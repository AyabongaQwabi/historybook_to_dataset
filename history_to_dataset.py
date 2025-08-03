import jsonlines
import requests
import re
import json
import time
import logging
from tqdm import tqdm
import PyPDF2
from datetime import datetime, timedelta
import psutil
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import textacy.preprocessing as preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy for named entity recognition and sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Default historical keywords - users can modify this set in the config section
DEFAULT_HISTORICAL_KEYWORDS = {
    'war', 'battle', 'conflict', 'treaty', 'empire', 'kingdom', 'dynasty', 'ruler', 
    'king', 'queen', 'emperor', 'chief', 'leader', 'revolution', 'independence',
    'colonial', 'colony', 'settlement', 'migration', 'civilization', 'culture',
    'tradition', 'ritual', 'ceremony', 'religion', 'missionary', 'trade',
    'conquest', 'invasion', 'rebellion', 'uprising', 'movement', 'reform',
    'ancient', 'medieval', 'renaissance', 'industrial', 'modern', 'contemporary',
    'century', 'decade', 'era', 'period', 'age', 'epoch'
}

def load_custom_keywords(keywords_file="keywords.txt"):
    """Load custom historical keywords from a file if it exists."""
    if os.path.exists(keywords_file):
        try:
            with open(keywords_file, 'r', encoding='utf-8') as f:
                custom_keywords = set(line.strip().lower() for line in f if line.strip())
            print(f"Loaded {len(custom_keywords)} custom keywords from {keywords_file}")
            logger.info(f"Loaded {len(custom_keywords)} custom keywords from {keywords_file}")
            return custom_keywords
        except Exception as e:
            print(f"Error loading keywords file: {e}. Using default keywords.")
            logger.warning(f"Error loading keywords file: {e}. Using default keywords.")
    return DEFAULT_HISTORICAL_KEYWORDS

def check_ollama_health():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.ok:
            print("Ollama server is running")
            logger.info("Ollama server is running")
            return True
        else:
            print(f"Ollama server error: {response.text}")
            logger.error(f"Ollama server responded with error: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Ollama server not reachable: {str(e)}")
        logger.error(f"Ollama server not reachable: {str(e)}")
        return False

def log_system_metrics(chunk_index):
    """Log CPU, memory, and GPU usage every 10 chunks."""
    if chunk_index % 10 != 0:
        return
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    mem_usage = mem.used / (1024 ** 3)  # GB
    mem_total = mem.total / (1024 ** 3)  # GB
    print(f"System metrics: CPU {cpu_percent:.1f}%, Memory {mem_usage:.1f}/{mem_total:.1f} GB")
    logger.info(f"System metrics: CPU {cpu_percent:.1f}%, Memory {mem_usage:.1f}/{mem_total:.1f} GB")
    try:
        import pynvml
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(device)
        gpu_usage = gpu_mem.used / (1024 ** 3)  # GB
        gpu_total = gpu_mem.total / (1024 ** 3)  # GB
        print(f"GPU metrics: VRAM {gpu_usage:.1f}/{gpu_total:.1f} GB")
        logger.info(f"GPU metrics: VRAM {gpu_usage:.1f}/{gpu_total:.1f} GB")
    except Exception as e:
        print(f"Could not retrieve GPU metrics: {str(e)}")
        logger.warning(f"Could not retrieve GPU metrics: {str(e)}")

def read_pdf_text(pdf_path):
    """Read text from a PDF file."""
    start_time = time.time()
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        print(f"Read PDF: {len(text)} characters in {time.time() - start_time:.2f}s")
        logger.info(f"Read PDF: {len(text)} characters in {time.time() - start_time:.2f}s")
        
        # Check for historical keywords
        historical_keywords = load_custom_keywords()
        found_keywords = [k for k in historical_keywords if k in text.lower()]
        print(f"Historical keywords found: {found_keywords[:10]}...")  # Show first 10
        logger.info(f"Historical keywords found: {found_keywords[:10]}...")
        if not found_keywords:
            print("Warning: PDF text may lack sufficient historical content for Q&A generation")
            logger.warning("PDF text may lack sufficient historical content for Q&A generation")
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""

def chunk_text(text, max_chars=800, overlap=200, min_chunks=3):
    """Split text into semantically meaningful chunks."""
    historical_keywords = load_custom_keywords()
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    current_length = 0
    
    # Try splitting by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_doc = nlp(para)
        has_historical_content = any(keyword in para.lower() for keyword in historical_keywords) or \
                                any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in para_doc.ents)
        if not has_historical_content:
            print(f"Discarded paragraph (no historical content): {para[:100]}...")
            logger.debug(f"Discarded paragraph (no historical content): {para[:100]}...")
            continue
        para_length = len(para)
        
        if current_length + para_length <= max_chars:
            current_chunk += para + "\n\n"
            current_length += para_length + 2
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para[-overlap:] if len(para) > overlap else para
            current_length = len(current_chunk)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If too few chunks, split by sentences
    if len(chunks) < min_chunks and text:
        chunks = []
        current_chunk = ""
        current_length = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sent_length = len(sent_text)
            has_historical_content = any(keyword in sent_text.lower() for keyword in historical_keywords) or \
                                    any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in sent.ents)
            if not has_historical_content:
                print(f"Discarded sentence (no historical content): {sent_text[:100]}...")
                logger.debug(f"Discarded sentence (no historical content): {sent_text[:100]}...")
                continue
            if current_length + sent_length <= max_chars:
                current_chunk += sent_text + " "
                current_length += sent_length + 1
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent_text[-overlap:] if len(sent_text) > overlap else sent_text
                current_length = len(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    print(f"Split text into {len(chunks)} chunks (text length: {len(text)} chars)")
    logger.info(f"Split text into {len(chunks)} chunks (text length: {len(text)} chars)")
    return chunks

def normalize_text(text):
    """Normalize text to handle case sensitivity and clean up."""
    return preprocessing.normalize.unicode(text.lower())

def is_historical_qa(question, answer):
    """Check if Q&A pair is historical using named entity recognition or keywords."""
    doc_q = nlp(question)
    doc_a = nlp(answer)
    historical_keywords = load_custom_keywords()
    has_entities = any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in doc_q.ents + doc_a.ents)
    has_keywords = any(keyword in question.lower() or keyword in answer.lower() for keyword in historical_keywords)
    return has_entities or has_keywords

def deduplicate_qa_pairs(pairs):
    """Remove duplicate Q&A pairs based on semantic similarity."""
    if not pairs:
        return pairs
    
    texts = [pair["instruction"] + " " + pair["output"] for pair in pairs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    keep_indices = []
    for i in range(len(pairs)):
        if i not in keep_indices:
            for j in range(i + 1, len(pairs)):
                if similarity_matrix[i][j] > 0.9:  # Threshold for near-duplicates
                    continue
                keep_indices.append(j)
            keep_indices.append(i)
    
    deduped_pairs = [pairs[i] for i in sorted(set(keep_indices))]
    print(f"Deduplicated {len(pairs)} to {len(deduped_pairs)} Q&A pairs")
    logger.info(f"Deduplicated {len(pairs)} to {len(deduped_pairs)} Q&A pairs")
    return deduped_pairs[:12]  # Limit to 12 pairs

def fix_json_string(json_str):
    """Attempt to fix common JSON errors (trailing commas, invalid escapes)."""
    try:
        # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        # Fix invalid escapes
        json_str = re.sub(r'\\[^\\bfnrtu"]', r'\\', json_str)
        return json_str
    except Exception as e:
        print(f"Failed to fix JSON string: {str(e)}")
        logger.error(f"Failed to fix JSON string: {str(e)}")
        return json_str

def extract_relevant_input(chunk, question, full_text):
    """Extract relevant sentences from the chunk or full text for the input field."""
    historical_keywords = load_custom_keywords()
    doc = nlp(chunk)
    relevant_sentences = []
    question_keywords = set(normalize_text(question).split()) & historical_keywords
    
    # First, try to find relevant sentences in the chunk
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if any(keyword in normalize_text(sent_text) for keyword in question_keywords) or \
           any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in sent.ents):
            relevant_sentences.append(sent_text)
    
    # If no relevant sentences found, search the full text
    if not relevant_sentences:
        doc_full = nlp(full_text)
        for sent in doc_full.sents:
            sent_text = sent.text.strip()
            if any(keyword in normalize_text(sent_text) for keyword in question_keywords) or \
               any(ent.label_ in {'PERSON', 'GPE', 'ORG', 'DATE', 'EVENT'} for ent in sent.ents):
                relevant_sentences.append(sent_text)
                if len(' '.join(relevant_sentences)) >= 800:
                    break
    
    input_text = ' '.join(relevant_sentences)[:800]
    if not input_text:
        input_text = chunk[:800]  # Fallback to original chunk if no relevant sentences found
    return input_text

def extract_json_array(text, chunk, full_text):
    """Extract JSON array from text, handling malformed cases."""
    exclude_patterns = [
        r'who.*wrote', r'who.*authored', r'what.*title', r'what.*published', 
        r'what.*topic.*passage', r'what.*debate', r'what.*focus.*research',
        r'who.*supervisor', r'what.*permits', r'what.*financial', r'what.*table of contents',
        r'who.*mentioned', r'what.*orthography', r'who.*provided', r'what.*list',
        r'what.*abbreviations', r'who.*assisted', r'who.*intellectually',
        r'what.*mean', r'define\s+', r'what.*structure'
    ]
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found")
        json_str = text[start:end]
        json_str = fix_json_string(json_str)
        parsed = json.loads(json_str)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError("Invalid JSON structure: not a list of dictionaries")
        
        filtered_pairs = []
        for item in parsed:
            question = normalize_text(item["instruction"])
            answer = normalize_text(item["output"])
            if (not any(re.search(pattern, question) for pattern in exclude_patterns) and 
                len(answer) >= 100 and is_historical_qa(question, answer)):
                item["input"] = extract_relevant_input(chunk, question, full_text)
                item["instruction"] = question.capitalize()
                item["output"] = answer
                filtered_pairs.append(item)
            else:
                print(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-historical or too short)")
                logger.debug(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-historical or too short)")
        
        print(f"Parsed {len(filtered_pairs)} valid JSON Q&A pairs")
        logger.info(f"Parsed {len(filtered_pairs)} valid JSON Q&A pairs")
        return deduplicate_qa_pairs(filtered_pairs)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing failed: {str(e)}. Non-JSON response:\n{text[:500]}...")
        logger.warning(f"JSON parsing failed: {str(e)}. Non-JSON response:\n{text[:500]}...")
        return parse_qa_pairs(text, chunk, full_text)

def parse_qa_pairs(text, chunk, full_text):
    """Parse Q: A: pairs from non-JSON response, focusing on historical content."""
    qa_pairs = []
    qa_pattern = re.compile(r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|$)', re.DOTALL)
    matches = qa_pattern.findall(text)
    
    exclude_patterns = [
        r'who.*wrote', r'who.*authored', r'what.*title', r'what.*published', 
        r'what.*topic.*passage', r'what.*debate', r'what.*focus.*research',
        r'who.*supervisor', r'what.*permits', r'what.*financial', r'what.*table of contents',
        r'who.*mentioned', r'what.*orthography', r'who.*provided', r'what.*list',
        r'what.*abbreviations', r'who.*assisted', r'who.*intellectually',
        r'what.*mean', r'define\s+', r'what.*structure'
    ]
    
    for question, answer in matches:
        question = normalize_text(question.strip())
        answer = normalize_text(answer.strip())
        if (not any(re.search(pattern, question) for pattern in exclude_patterns) and 
            len(answer) >= 100 and is_historical_qa(question, answer)):
            qa_pairs.append({
                "instruction": question.capitalize(),
                "input": extract_relevant_input(chunk, question, full_text),
                "output": answer
            })
            print(f"Accepted Q&A pair: Q: {question.capitalize()}, A: {answer[:50]}...")
            logger.info(f"Accepted Q&A pair: Q: {question.capitalize()}, A: {answer[:50]}...")
        else:
            print(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-historical or too short)")
            logger.debug(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-historical or too short)")
    
    if not qa_pairs:
        print(f"No valid Q&A pairs found in output for chunk:\n{chunk[:500]}...")
        logger.warning(f"No valid Q&A pairs found in output for chunk:\n{chunk[:500]}...")
    else:
        print(f"Extracted {len(qa_pairs)} Q&A pairs via fallback")
        logger.info(f"Extracted {len(qa_pairs)} Q&A pairs via fallback")
    
    return deduplicate_qa_pairs(qa_pairs)

def generate_questions_answers(chunk, full_text, model_name="llama3.1", max_retries=5):
    """Generate 1–12 Q&A pairs about historical figures, events, and cultural items."""
    print(f"Processing chunk (first 200 chars): {chunk[:200]}...")
    logger.info(f"Processing chunk (first 200 chars): {chunk[:200]}...")
    start_time = time.time()
    prompt = f"""
You are a historian tasked with generating 1–12 high-quality question-answer pairs from a given text passage for a fine-tuned question-answering model. Your output MUST be a valid JSON array containing 1–12 objects, each with the fields "instruction" (the question), "input" (the specific sentences or phrases from the passage that directly relate to the question and answer), and "output" (the answer). Do NOT include any text outside the JSON array (e.g., explanations, headings, Q: A: pairs). Non-JSON output will be discarded.

Focus EXCLUSIVELY on questions about historical figures (e.g., rulers, leaders, warriors, scholars, missionaries), events (e.g., wars, battles, treaties, revolutions, movements), and cultural items or practices (e.g., traditions, rituals, ceremonies, literature, art, customs), including minor or less prominent ones. For the "input" field, include only the sentences or phrases from the passage that directly support the question and answer, ensuring the input is complete and meaningful. Answers must be at least 100 characters, include specific details (e.g., quotes, names, dates, roles, significance), and explicitly cite sources mentioned in the passage or note if no source is provided. Avoid non-historical topics (e.g., authorship, publication, supervisors, funding, table of contents, orthography, abbreviations, acknowledgments, definitions, structure). If the passage is short or lacks sufficient content, generate at least 1 high-quality pair, prioritizing historical relevance.

Ensure answers are detailed, accurate, and contextually rich, drawing directly from the passage. Verify relationships and use primary sources or interviews cited in the passage for accuracy. If the passage lacks specific details, do NOT generate Q&A pairs based on external knowledge.

Example output:
[
  {{"instruction": "What was the significance of the Battle of Hastings?", "input": "The Battle of Hastings in 1066 marked the Norman conquest of England.", "output": "The Battle of Hastings in 1066 was a pivotal moment that marked the Norman conquest of England, fundamentally changing English society, language, and governance under William the Conqueror's rule."}},
  {{"instruction": "How did medieval guilds influence trade?", "input": "Medieval guilds controlled trade practices and maintained quality standards in cities.", "output": "Medieval guilds were powerful organizations that controlled trade practices, maintained quality standards, and regulated prices in cities, effectively shaping the economic landscape of medieval Europe through their monopolistic control over crafts and commerce."}}
]

Generate 1–12 high-quality question-answer pairs based on the following passage:
\"\"\"{chunk}\"\"\"
"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.5
                },
                timeout=180
            )
            if not response.ok:
                print(f"LLaMA request failed (attempt {attempt+1}/{max_retries}): {response.text}")
                logger.error(f"LLaMA request failed (attempt {attempt+1}/{max_retries}): {response.text}")
                time.sleep(5)
                continue

            raw = response.json()["response"]
            with open("raw_responses.log", "a") as f:
                f.write(f"Chunk response at {datetime.now()}:\n{raw}\n{'='*50}\n")
            print(f"Response time: {time.time() - start_time:.2f}s")
            logger.info(f"Response time: {time.time() - start_time:.2f}s")
            return extract_json_array(raw, chunk, full_text)
        except Exception as e:
            print(f"Error processing chunk (attempt {attempt+1}/{max_retries}): {str(e)}")
            logger.error(f"Error processing chunk (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to process chunk after {max_retries} attempts:\n{chunk[:500]}...")
                logger.error(f"Failed to process chunk after {max_retries} attempts:\n{chunk[:500]}...")
                return []
            time.sleep(5)
    return []

def process_chunk(i, chunk, full_text, model_name):
    """Helper function to process a single chunk and return results."""
    log_system_metrics(i)
    items = generate_questions_answers(chunk, full_text, model_name)
    return i, chunk, items

def generate_dataset_from_pdf(pdf_path, output_path, model_name="llama3.1", start_chunk=0, temp_path=None, checkpoint_path="checkpoint.json", max_workers=4):
    """Generate a dataset from a PDF and save to JSONL, processing chunks in parallel."""
    if not check_ollama_health():
        print("Aborting: Ollama server not available")
        logger.error("Aborting: Ollama server not available")
        return

    if temp_path is None:
        bookname = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_path = f"temp_{bookname}.jsonl"
    
    start_time = time.time()
    full_text = read_pdf_text(pdf_path)
    print(f"First 500 chars of full text: {full_text[:500]}")
    logger.info(f"First 500 chars of full text: {full_text[:500]}")
    if len(full_text) < 200:
        print(f"Warning: Input text is too short ({len(full_text)} chars). Consider augmenting with additional sources.")
        logger.warning(f"Input text is too short ({len(full_text)} chars). Consider augmenting with additional sources.")
    
    chunks = chunk_text(full_text)
    
    if start_chunk < 0 or start_chunk >= len(chunks):
        print(f"Invalid start_chunk {start_chunk}; must be between 0 and {len(chunks)-1}")
        logger.error(f"Invalid start_chunk {start_chunk}; must be between 0 and {len(chunks)-1}")
        return

    all_data = []
    stats = {"successful_chunks": 0, "failed_chunks": 0, "total_pairs": 0}
    failed_chunks = []
    lock = threading.Lock()

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        all_data = checkpoint.get("data", [])
        stats = checkpoint.get("stats", {"successful_chunks": 0, "failed_chunks": 0, "total_pairs": 0})
        failed_chunks = checkpoint.get("failed_chunks", [])
        checkpoint_last = checkpoint.get("last_chunk", -1)
        if checkpoint_last >= start_chunk:
            start_chunk = checkpoint_last + 1
            print(f"Resumed from checkpoint: {len(all_data)} pairs, starting at chunk {start_chunk}")
            logger.info(f"Resumed from checkpoint: {len(all_data)} pairs, starting at chunk {start_chunk}")
        else:
            print(f"Ignoring checkpoint (last_chunk {checkpoint_last} < start_chunk {start_chunk})")
            logger.info(f"Ignoring checkpoint (last_chunk {checkpoint_last} < start_chunk {start_chunk})")

    try:
        with tqdm(total=len(chunks), initial=start_chunk, desc="Processing chunks", unit="chunk") as pbar:
            for batch_start in range(start_chunk, len(chunks), max_workers):
                batch = [(i, chunk) for i, chunk in enumerate(chunks[batch_start:batch_start + max_workers], start=batch_start)]
                futures = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for i, chunk in batch:
                        futures.append(executor.submit(process_chunk, i, chunk, full_text, model_name))
                    
                    for future in as_completed(futures):
                        try:
                            i, chunk, items = future.result()
                            with lock:
                                if items:
                                    all_data.extend(items)
                                    stats["successful_chunks"] += 1
                                    stats["total_pairs"] += len(items)
                                    print(f"Chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                                    logger.info(f"Chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                                    if stats["successful_chunks"] % 3 == 0:
                                        with jsonlines.open(temp_path, mode='w') as writer:
                                            writer.write_all(all_data)
                                        print(f"Saved partial results to {temp_path}")
                                        logger.info(f"Saved partial results to {temp_path}")
                                else:
                                    stats["failed_chunks"] += 1
                                    failed_chunks.append((i, chunk))
                                    print(f"Chunk {i}/{len(chunks)}: No Q&A pairs generated, added to failed_chunks")
                                    logger.warning(f"Chunk {i}/{len(chunks)}: No Q&A pairs generated, added to failed_chunks")
                                
                                with open(checkpoint_path, 'w') as f:
                                    json.dump({
                                        "data": all_data,
                                        "stats": stats,
                                        "last_chunk": i,
                                        "failed_chunks": failed_chunks
                                    }, f)
                                
                                elapsed = time.time() - start_time
                                processed_chunks = i - start_chunk + 1 if i >= start_chunk else 1
                                avg_time_per_chunk = elapsed / processed_chunks
                                remaining_chunks = len(chunks) - i
                                eta_seconds = remaining_chunks * avg_time_per_chunk
                                eta = str(timedelta(seconds=int(eta_seconds)))
                                pbar.set_postfix({"ETA": eta, "Pairs": stats["total_pairs"]})
                                pbar.update(1)
                        except Exception as e:
                            print(f"Error processing future for chunk {i}: {str(e)}")
                            logger.error(f"Error processing future for chunk {i}: {str(e)}")
                
                time.sleep(2)

        if failed_chunks:
            print(f"Retrying {len(failed_chunks)} failed chunks sequentially")
            logger.info(f"Retrying {len(failed_chunks)} failed chunks sequentially")
            for i, chunk in failed_chunks:
                log_system_metrics(i)
                print(f"Retrying chunk {i}/{len(chunks)}")
                logger.info(f"Retrying chunk {i}/{len(chunks)}")
                items = generate_questions_answers(chunk, full_text, model_name)
                with lock:
                    if items:
                        all_data.extend(items)
                        stats["successful_chunks"] += 1
                        stats["total_pairs"] += len(items)
                        print(f"Retry chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                        logger.info(f"Retry chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                        if stats["successful_chunks"] % 3 == 0:
                            with jsonlines.open(temp_path, mode='w') as writer:
                                writer.write_all(all_data)
                            print(f"Saved partial results to {temp_path}")
                            logger.info(f"Saved partial results to {temp_path}")
                    else:
                        print(f"Retry chunk {i}/{len(chunks)}: No Q&A pairs generated:\n{chunk[:500]}...")
                        logger.warning(f"Retry chunk {i}/{len(chunks)}: No Q&A pairs generated:\n{chunk[:500]}...")
                    with open(checkpoint_path, 'w') as f:
                        json.dump({
                            "data": all_data,
                            "stats": stats,
                            "last_chunk": i,
                            "failed_chunks": failed_chunks
                        }, f)

    except Exception as e:
        print(f"Processing interrupted: {str(e)}. Saving partial results")
        logger.error(f"Processing interrupted: {str(e)}. Saving partial results")
        with lock:
            if all_data:
                with jsonlines.open(temp_path, mode='w') as writer:
                    writer.write_all(all_data)
                print(f"Saved partial results to {temp_path}")
                logger.info(f"Saved partial results to {temp_path}")

    print("Generated Q&A pairs (sample of up to 10):")
    logger.info("Generated Q&A pairs (sample of up to 10):")
    for i, item in enumerate(all_data[:10], 1):
        print(f"Pair {i}:")
        print(f"  Instruction: {item['instruction']}")
        print(f"  Input: {item['input'][:200]}...")
        print(f"  Output: {item['output'][:100]}...")
        print(f"  Input length: {len(item['input'])} chars")
        logger.info(f"Pair {i}:")
        logger.info(f"  Instruction: {item['instruction']}")
        logger.info(f"  Input: {item['input'][:200]}...")
        logger.info(f"  Output: {item['output'][:100]}...")
        logger.info(f"  Input length: {len(item['input'])} chars")

    if all_data:
        with jsonlines.open(output_path, mode='w') as writer:
            writer.write_all(all_data)
        print(f"Dataset written to {output_path} with {len(all_data)} items")
        logger.info(f"Dataset written to {output_path} with {len(all_data)} items")
    else:
        print(f"No data written to {output_path}: No Q&A pairs generated")
        logger.warning(f"No data written to {output_path}: No Q&A pairs generated")

    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"Processing complete in {elapsed_time}")
    print(f"Summary: {stats['successful_chunks']} successful chunks, "
          f"{stats['failed_chunks']} failed chunks, {stats['total_pairs']} total Q&A pairs")
    logger.info(f"Processing complete in {elapsed_time}")
    logger.info(f"Summary: {stats['successful_chunks']} successful chunks, "
                f"{stats['failed_chunks']} failed chunks, {stats['total_pairs']} total Q&A pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q&A dataset from PDF")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("output_path", help="Path for the output JSONL file")
    parser.add_argument("--start-chunk", type=int, default=0, help="Starting chunk index (0-based, default 0)")
    parser.add_argument("--model-name", type=str, default="llama3.1", help="Ollama model name (e.g., llama3.1, mistral)")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers (default 4)")
    args = parser.parse_args()

    generate_dataset_from_pdf(args.pdf_path, args.output_path, model_name=args.model_name, start_chunk=args.start_chunk, max_workers=args.max_workers)
