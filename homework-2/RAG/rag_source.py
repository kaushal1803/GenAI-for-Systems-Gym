"""
This file implements the RAG algorithm. Its functions are called by RAG_application.ipynb


Key Features:

1. PC and Function Mapping: Matches each PC to its assembly instructions and source code function.

2. Embedding-Based Retrieval: Computes embeddings for workload and policy descriptions to support 
   similarity search.

3. Query Processing and Ranking:
   - Extracts PCs and memory addresses from user queries.
   - Filters and ranks traces using keyword matching and semantic similarity.
   - Selects the most relevant traces for the LLM’s context window.

4. Data Extraction for LLM:
   - Aggregates cache statistics, including miss rates, reuse distances, and eviction patterns.
   - Retrieves function names and assembly snippets for identified PCs.
   - Returns structured responses that explain cache behavior within replacement policies.

This system retrieves both low-level access details and high-level policy performance trends.
"""



import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

def prepare_data_for_rag(processed_data, model, max_assembly_lines=5):
    """
    Connect each PC in the preprocessed data to its corresponding assembly code and source function. 
    Also embed each trace's metadata to enable similarity search with the user query.
    """
    for trace_id, content in processed_data.items():
        df = content["data_frame"]
        
        # Group by function_name
        df_grouped = df.groupby('function_name', dropna=False)
        
        chunks = []
        
        for func_name, group in df_grouped:
            if pd.isna(func_name):
                func_name_str = "Unknown_Function"
            else:
                func_name_str = func_name
            
            # Collect assembly snippets
            assembly_samples = []
            for idx, row in group.head(max_assembly_lines).iterrows():
                assembly_lines = row.get("assembly_code", "")
                assembly_excerpt = "\n".join(assembly_lines.split("\n")[:2])
                assembly_samples.append(assembly_excerpt)
            
            assembly_text = "\n".join(assembly_samples)
            
            # Only function name, no function code
            chunk_text = (
                f"Source Function: {func_name_str}\n\n"
                f"Assembly Snippet:\n{assembly_text}"
            )
            
            chunks.append({
                "function_name": func_name_str,
                "text": chunk_text
            })
        
        # Compute embeddings
        for c in chunks:
            c["embedding"] = model.encode(c["text"], convert_to_tensor=True)
        
        processed_data[trace_id]["chunks"] = chunks

    return processed_data


def filter_and_rank_traces(query, trace_metadata, model):

    """ 
    Performs trace selection based on the user query to limit the amount of data sent to the LLM's context window. 
    Selects the three most relevant traces based on keyword searching and semantic similarity between the user query and each trace's metadata.

    """
    workload_keywords = ["bzip", "libq", "ibm", "milc"]
    policy_keywords = ["imitation", "learned", "learning", "PARROT", "LRU", "least recently used", "MLP", "perception", "RNN"]

    query_lower = query.lower()

    # Workload filter
    found_workload_keywords = [kw for kw in workload_keywords if kw in query_lower]
    if found_workload_keywords:
        traces_after_workload_filter = {
            trace_id: meta for trace_id, meta in trace_metadata.items()
            if any(kw in meta['workload_description'].lower() for kw in found_workload_keywords)
        }
    else:
        traces_after_workload_filter = trace_metadata

    # Policy filter
    found_policy_keywords = [kw for kw in policy_keywords if kw in query_lower]
    if found_policy_keywords:
        traces_after_policy_filter = {
            trace_id: meta for trace_id, meta in traces_after_workload_filter.items()
            if any(kw.lower() in meta['policy_description'].lower() for kw in found_policy_keywords)
        }
    else:
        traces_after_policy_filter = traces_after_workload_filter

    if not traces_after_policy_filter:
        return []

    # Semantic similarity
    metadata_texts = {
        trace_id: f"{meta['policy_description']} {meta['workload_description']}" 
        for trace_id, meta in traces_after_policy_filter.items()
    }

    query_embedding = model.encode(query, convert_to_tensor=True)
    trace_embeddings = {trace_id: model.encode(text, convert_to_tensor=True) for trace_id, text in metadata_texts.items()}
    
    similarities = {trace_id: util.pytorch_cos_sim(query_embedding, embedding).item() for trace_id, embedding in trace_embeddings.items()}
    sorted_traces = sorted(similarities, key=similarities.get, reverse=True)
    top_traces = sorted_traces[:3]

    return top_traces

def extract_pc_and_address(query):

    """Uses regex to find the PC and address if present in the user query"""

    hex_pattern = r'0x[0-9a-fA-F]+'
    hex_matches = re.findall(hex_pattern, query.lower())
    
    if len(hex_matches) >= 2:
        return hex_matches[0], hex_matches[1]
    elif len(hex_matches) == 1:
        return hex_matches[0], None
    else:
        return None, None

def process_query(query, processed_data, embedding_model):

    """ High-level function that is called by the RAG interface"""

    # Determine relevant traces
    trace_metadata = {
        trace_id: {
            "policy_description": data["description"].split("\n")[0].replace("Replacement Policy: ", ""),
            "workload_description": data["description"].split("\n")[1].replace("Workload: ", "")
        } for trace_id, data in processed_data.items()
    }
    relevant_traces = filter_and_rank_traces(query, trace_metadata, embedding_model)

    # Optional second-level retrieval refinement
    if relevant_traces:
        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        refined_scores = {}
        for trace_id in relevant_traces:
            chunks = processed_data[trace_id].get("chunks", [])
            if not chunks:
                refined_scores[trace_id] = 0
                continue
            best_score = max(util.pytorch_cos_sim(query_embedding, c["embedding"]).item() for c in chunks)
            refined_scores[trace_id] = best_score
        relevant_traces = sorted(refined_scores.keys(), key=lambda x: refined_scores[x], reverse=True)[:3]

    pc, address = extract_pc_and_address(query)

    # Data Gathering
    workloads_seen = set()
    policies_seen = set()
    info_list = []

    def compute_pc_stats(df, pc):
        """Function to compute PC-level statistics so the RAG can answer queries about all the data for a given PC"""

        pc_rows = df[df['program_counter'] == pc]
        if pc_rows.empty:
            return None
        num_appearances = len(pc_rows)
        miss_rate = pc_rows['evict'].value_counts(normalize=True).get('Cache Miss', 0)*100
        avg_reuse_dist_accessed = pd.to_numeric(pc_rows['accessed_address_reuse_distance'], errors='coerce').mean()
        missed_rows = pc_rows[pc_rows['evict'] == "Cache Miss"].copy()
        avg_reuse_dist_evicted = pd.to_numeric(missed_rows['evicted_address_reuse_distance'], errors='coerce').mean()
        missed_rows['evicted_address_reuse_distance'] = pd.to_numeric(missed_rows['evicted_address_reuse_distance'], errors='coerce')
        missed_rows['accessed_address_reuse_distance'] = pd.to_numeric(missed_rows['accessed_address_reuse_distance'], errors='coerce')
        bad_evictions = missed_rows[missed_rows['evicted_address_reuse_distance'] < missed_rows['accessed_address_reuse_distance']]
        bad_percent = (len(bad_evictions)/len(missed_rows))*100 if len(missed_rows)>0 else 0
        return {
            "appearances": num_appearances,
            "miss_rate": miss_rate,
            "avg_accessed_reuse": avg_reuse_dist_accessed,
            "avg_evicted_reuse": avg_reuse_dist_evicted,
            "bad_evictions": bad_percent
        }

    # If we have a PC, we might want to retrieve assembly code and function name from a representative row.
    assembly_code_snippet = None
    function_name = None

    for trace_id in relevant_traces:
        df = processed_data[trace_id]["data_frame"]
        policy_desc = trace_metadata[trace_id]["policy_description"]
        workload_desc = trace_metadata[trace_id]["workload_description"]

        workloads_seen.add(workload_desc)
        policies_seen.add(policy_desc)

        if pc and address:
            # Specific row
            row = df[(df['program_counter'] == pc) & (df['memory_address'] == address)]
            if not row.empty:
                r = row.iloc[0]
                # Store PC+address info
                info_list.append({
                    "type": "pc_address",
                    "policy": policy_desc,
                    "workload": workload_desc,
                    "pc": pc,
                    "address": address,
                    "evict": r['evict'],
                    "evicted_addr": r['evicted_address'],
                    "evicted_reuse": r['evicted_address_reuse_distance'],
                    "accessed_reuse": r['accessed_address_reuse_distance']
                })
                # Save assembly code and function name if not already set
                if assembly_code_snippet is None:
                    assembly_code_snippet = r.get("assembly_code", "")
                if function_name is None:
                    function_name = r.get("function_name", "Unknown_Function")

        elif pc:
            # Just PC stats
            stats = compute_pc_stats(df, pc)
            if stats:
                info_list.append({
                    "type": "pc_only",
                    "policy": policy_desc,
                    "workload": workload_desc,
                    "pc": pc,
                    "stats": stats
                })
                # For code snippet and function name
                if assembly_code_snippet is None:
                    # Take a representative row
                    rep_row = df[df['program_counter'] == pc].iloc[0]
                    assembly_code_snippet = rep_row.get("assembly_code", "")
                    function_name = rep_row.get("function_name", "Unknown_Function")

        else:
            # No PC or address
            info_list.append({
                "type": "no_pc",
                "policy": policy_desc,
                "workload": workload_desc,
                "metadata": processed_data[trace_id]["metadata"]
            })

    # Build the response
    response_text = "You are an expert in computer architecture and your job is to answer questions given data from cache traces. "
    response_text += "Base your response on the following data and your knowledge of computer architecture.\n\n"

    # Print workloads once
    if workloads_seen:
        response_text += "Workloads involved:\n"
        for w in workloads_seen:
            response_text += f"- {w}\n"
        response_text += "\n"
    # Print policies once
    if policies_seen:
        response_text += "Policies involved:\n"
        for p in policies_seen:
            response_text += f"- {p}\n"
        response_text += "\n"

    # If we have a PC, print assembly code and function name once
    if pc and assembly_code_snippet is not None and function_name is not None:
        response_text += f"Assembly code snippet for PC {pc} (representative instructions):\n{assembly_code_snippet}\n\n"
        response_text += f"Source Function: {function_name}\n\n"

    # Now present data depending on the query type
    if pc and address:
        # Show each policy's hit/miss result
        for info in info_list:
            if info["type"] == "pc_address":
                response_text += (
                    f"For policy {info['policy'].split(':')[0]} on workload {info['workload'].split(':')[0]} at PC {info['pc']} and address {info['address']}:\n"
                    f"Cache result: {info['evict']}\n"
                )
                if info['evict'] == "Cache Miss":
                    response_text += (
                        f"Evicted address: {info['evicted_addr']} (needed again in {info['evicted_reuse']} accesses), "
                        f"Inserted address needed again in {info['accessed_reuse']} accesses.\n\n"
                    )
                else:
                    response_text += "\n"

    elif pc:
        # Show stats for each policy
        for info in info_list:
            if info["type"] == "pc_only":
                s = info["stats"]
                response_text += (
                    f"For policy {info['policy'].split(':')[0]} on workload {info['workload'].split(':')[0]} at PC {info['pc']}:\n"
                    f"- Miss Rate: {s['miss_rate']:.2f}%\n"
                    f"- Avg Accessed Reuse Distance: {s['avg_accessed_reuse']:.2f}\n"
                    f"- Avg Evicted Reuse Distance: {s['avg_evicted_reuse']:.2f}\n"
                    f"- Incorrect Evictions: {s['bad_evictions']:.2f}%\n\n"
                )

    else:
        # No PC/address: Just show metadata for each trace
        # The metadata might be large; show it once per unique (workload,policy) combo
        shown_combos = set()
        for info in info_list:
            combo = (info["workload"], info["policy"])
            if combo not in shown_combos:
                shown_combos.add(combo)
                response_text += f"For {info['policy'].split(':')[0]} on {info['workload'].split(':')[0]}:\n{info['metadata']}\n\n"

    # Finally, restate the query
    response_text += f"Answer the following question: {query}"

    return response_text
