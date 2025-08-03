# Changelog

## Version 1.0.0 (Initial Release)

### Features
- PDF text extraction and intelligent chunking
- AI-powered Q&A pair generation using Ollama
- Customizable historical keyword filtering
- Parallel processing with configurable worker threads
- Checkpoint system for resuming interrupted processing
- Semantic deduplication of Q&A pairs
- Comprehensive logging and system monitoring
- Support for multiple AI models through Ollama

### Supported Models
- Llama 3.1 (default)
- Mistral
- Any Ollama-compatible model

### Output Format
- JSONL format compatible with AI training frameworks
- Structured Q&A pairs with instruction, input, and output fields
