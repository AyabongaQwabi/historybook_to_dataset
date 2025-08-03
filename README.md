# History Book to Dataset Generator

A Python tool that converts history books (PDF format) into question-answer datasets suitable for training AI models. The tool uses natural language processing and AI to automatically generate high-quality Q&A pairs from historical texts.

## Features

- **PDF Processing**: Extracts text from PDF files and intelligently chunks it for processing
- **AI-Powered Q&A Generation**: Uses Ollama (local AI models) to generate contextual questions and answers
- **Historical Content Filtering**: Focuses on historical figures, events, and cultural practices
- **Customizable Keywords**: Configure domain-specific historical keywords
- **Parallel Processing**: Multi-threaded processing for faster dataset generation
- **Resume Capability**: Checkpoint system allows resuming interrupted processing
- **Deduplication**: Removes similar Q&A pairs to ensure dataset quality
- **Comprehensive Logging**: Detailed logs for monitoring and debugging

## Prerequisites

### 1. Ollama Installation

This tool requires Ollama to be installed and running on your system. Ollama provides local AI model inference.

#### Install Ollama:

**macOS/Linux:**
\`\`\`bash
curl -fsSL https://ollama.ai/install.sh | sh
\`\`\`

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

#### Start Ollama and Download a Model:
\`\`\`bash
# Start Ollama service
ollama serve

# In another terminal, download a model (e.g., Llama 3.1)
ollama pull llama3.1
\`\`\`

Verify Ollama is running by visiting `http://localhost:11434` in your browser.

### 2. Python Dependencies

Install the required Python packages:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. spaCy Language Model

Download the English language model for spaCy:

\`\`\`bash
python -m spacy download en_core_web_sm
\`\`\`

## Usage

### Basic Usage

The script requires two mandatory parameters: the input PDF file path and the output dataset file path.

\`\`\`bash
python history_to_dataset.py <input_pdf_path> <output_jsonl_path>
\`\`\`

**Examples:**
\`\`\`bash
# Process a history book and save to dataset
python history_to_dataset.py "my_history_book.pdf" "history_dataset.jsonl"

# Process with specific paths
python history_to_dataset.py "/path/to/book.pdf" "/path/to/output/dataset.jsonl"
\`\`\`

### Advanced Usage

\`\`\`bash
python history_to_dataset.py <input_pdf_path> <output_jsonl_path> \
    --model-name llama3.1 \
    --max-workers 4 \
    --start-chunk 0
\`\`\`

**Examples:**
\`\`\`bash
# Using different model and more workers
python history_to_dataset.py "book.pdf" "dataset.jsonl" --model-name mistral --max-workers 8

# Resume processing from chunk 50
python history_to_dataset.py "book.pdf" "dataset.jsonl" --start-chunk 50
\`\`\`

### Command Line Arguments

**Required Arguments:**
- `pdf_path`: Path to the input PDF file
- `output_path`: Path for the output JSONL dataset file

**Optional Arguments:**
- `--model-name`: Ollama model to use (default: llama3.1)
- `--max-workers`: Number of parallel processing threads (default: 4)
- `--start-chunk`: Starting chunk index for resuming processing (default: 0)

## Customizing Historical Keywords

The tool uses historical keywords to identify and filter relevant content. You can customize these keywords by editing the `keywords.txt` file.

### How to Add Custom Keywords:

1. Open `keywords.txt` in a text editor
2. Add one keyword per line (case-insensitive)
3. Lines starting with `#` are comments and will be ignored
4. Save the file

### Example Keywords for Different Historical Domains:

**African History:**
\`\`\`
xhosa
thembu
maqoma
ngqika
nongqause
phalo
cattle-killing
\`\`\`

**European History:**
\`\`\`
feudalism
crusades
reformation
enlightenment
napoleon
charlemagne
\`\`\`

**American History:**
\`\`\`
colonial
revolutionary
civil war
reconstruction
manifest destiny
new deal
\`\`\`

The tool will automatically load your custom keywords when processing documents.

## Output Format

The tool generates a JSONL (JSON Lines) file where each line contains a Q&A pair in the following format:

\`\`\`json
{
  "instruction": "What was the significance of the Battle of Hastings?",
  "input": "The Battle of Hastings in 1066 marked the Norman conquest of England.",
  "output": "The Battle of Hastings in 1066 was a pivotal moment that marked the Norman conquest of England, fundamentally changing English society, language, and governance under William the Conqueror's rule."
}
\`\`\`

## Configuration

### Model Selection

You can use different Ollama models by specifying the `--model-name` parameter:

\`\`\`bash
# Using Mistral
python history_to_dataset.py book.pdf dataset.jsonl --model-name mistral

# Using Llama 3.1 (default)
python history_to_dataset.py book.pdf dataset.jsonl --model-name llama3.1
\`\`\`

### Performance Tuning

- **max-workers**: Adjust based on your system's CPU cores and memory
- **GPU Usage**: The tool will automatically use GPU if available through Ollama
- **Memory**: Monitor system memory usage; reduce max-workers if needed

## Resuming Processing

If processing is interrupted, the tool automatically creates checkpoints. Simply run the same command again to resume from where it left off.

## Logging

The tool creates detailed logs in:
- `dataset_generation.log`: Main processing log
- `raw_responses.log`: Raw AI model responses for debugging

## Troubleshooting

### Common Issues:

1. **"Ollama server not reachable"**
   - Ensure Ollama is installed and running (`ollama serve`)
   - Check that port 11434 is not blocked

2. **"No historical keywords found"**
   - Review and update your `keywords.txt` file
   - Ensure your PDF contains historical content

3. **Low Q&A pair generation**
   - Check if your PDF text extraction is working properly
   - Verify that historical keywords match your content
   - Consider using a different AI model

4. **Memory issues**
   - Reduce the `--max-workers` parameter
   - Process smaller PDF files
   - Monitor system resources

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool.

## License

This project is open source. Please check the license file for details.
