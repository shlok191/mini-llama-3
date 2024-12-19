use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use pyo3::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

// Creating a Python wrapper for our tokenizer
#[pyclass(name = "MiniLlamaTokenizer")]
struct PyTokenizer {
    inner: BPETokenizer
}

// Implementing wrappers for the functions of the tokenizer
#[pymethods]
impl PyTokenizer {
    
    #[new]
    fn new(vocab_size: usize, iterations: usize) -> Self {
        PyTokenizer {
            inner: BPETokenizer::new(vocab_size, iterations)
        }
    }

    fn train(&mut self, texts: Vec<String>) -> PyResult<()> {
        self.inner.train(&texts)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn encode(&self, text: &str, max_length: Option<usize>) -> Vec<usize> {
        self.inner.encode(text, max_length)
    }

    fn decode(&self, token_ids: Vec<usize>) -> String {
        self.inner.decode(&token_ids)
    }
}

#[pymodule]
fn rust_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}


#[derive(Debug, Serialize, Deserialize)]
pub struct BPETokenizer {
    vocab: HashMap<String, usize>,
    merges: HashMap<(String, String), String>,
    special_tokens: HashSet<String>,
    vocab_size: usize,
    iterations: usize,
}

impl BPETokenizer {

    pub fn new(vocab_size: usize, iterations: usize) -> Self {

        // Defining our tokenizer with the needed data structures!
        let mut tokenizer = BPETokenizer {
            vocab: HashMap::new(),
            merges: HashMap::new(),
            special_tokens: HashSet::new(),
            vocab_size,
            iterations
        };

        // Initializing special tokens
        let special_tokens = vec![
            "<padding>",
            "<begin_of_sentence>",
            "<end_of_sentence>",
            "<unknown>",
            "</w>",
        ];

        // Adding our special tokens in first
        for token in special_tokens {

            tokenizer.special_tokens.insert(token.to_string());
            tokenizer.vocab.insert(token.to_string(), tokenizer.vocab.len());
        }

        // Initializing basic vocabulary with some standard ASCII characters!
        let basic_chars = "ABCDEFGHIJKLMNOPRQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/\"\\\'~";
        
        for c in basic_chars.chars() {
            
            // Add everything that is new
            if !tokenizer.vocab.contains_key(&c.to_string()) {
                tokenizer.vocab.insert(c.to_string(), tokenizer.vocab.len());
            }
        }

        println!("Finished initializing tokenizer!");
        // I like how Rust does not even have return statements, rad :)
        tokenizer
    }

    pub fn train(&mut self, texts: &[String]) -> Result<()> {
        
        println!("Starting training with {} texts", texts.len());
        
        // Phase 1: Building word frequencies
        println!("Building word frequencies...");

        // First progress bar for word frequency counting
        let word_freq_bar = ProgressBar::new(texts.len() as u64);
        
        // Creating a progress bar for this!
        word_freq_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} 🏴‍☠️ Word Frequencies [{elapsed_precise}] {bar:40.green/white} {pos:>7}/{len:7} {msg}")
                .unwrap()
        );

        // Keeps track of word frequencies
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        
        for text in texts {
            
            // Splits text into words and adds </w> to the end of each word for delimiting purposes
            let words = text.split_whitespace()
                .map(|w| format!("{}</w>", w));
            
            // Start adding word frequencies
            for word in words {
                *word_freqs.entry(word).or_insert(0) += 1;
            }

            word_freq_bar.inc(1);

            // Update message occasionally
            if word_freqs.len() % 1000 == 0 {
                word_freq_bar.set_message(format!("Unique words: {}", word_freqs.len()));
            }
        }
        
        word_freq_bar.finish_with_message(format!("Found {} unique words", word_freqs.len()));
        println!("Found {} unique words", word_freqs.len());

        // Phase 2: Merging tokens for iterations specified
        println!("Converting to character sequences...");

        // Creating a progress bar!
        let progress_bar = ProgressBar::new(self.iterations as u64);

        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} 🏴‍☠️ [{elapsed_precise}] {bar:40.green/white} {pos:>7}/{len:7} {msg}")
                .unwrap()
        );
        
        // Convert words to character sequences
        let mut word_splits: HashMap<String, usize> = word_freqs.iter()
            .map(|(word, &freq)| {
            
                let chars = word.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
                (chars, freq)
            })
            .collect();

        println!("Starting main training loop...");
        // Defining our training loop
        for i in 0..self.iterations {

            // Quit if we have reached maximum possible tokens
            if self.vocab.len() >= self.vocab_size {

                progress_bar.finish_with_message("Reached maximum vocabulary size! Arrr!");
                break;
            }

            // Getting pair frequencies in parallel
            let word_splits_arc = Arc::new(word_splits.clone());
            let pairs = self.get_pair_frequencies(&word_splits_arc);

            if pairs.is_empty() {
                progress_bar.finish_with_message("No more pairs to merge!");
                break;
            }

            // Finding the most frequently occuring token pair
            let best_pair = pairs.iter()
                .max_by_key(|&(_, count)| count)
                .ok_or_else(|| anyhow!("No pairs found"))?
                .0
                .clone();

            // Merging this best pair
            word_splits = self.merge_tokens(&best_pair, &word_splits);

            // Updating our vocabulary
            let merged = format!("{}{}", best_pair.0, best_pair.1);

            self.merges.insert(best_pair, merged.clone());
            self.vocab.insert(merged, self.vocab.len());

            if (i + 1) % 1000 == 0 {
                println!("Vocab size now: {}", self.vocab.len());
                progress_bar.set_message(format!("Vocab size: {}", self.vocab.len()));
            }

            progress_bar.inc(1);
        }

        progress_bar.finish();
        Ok(())
    }

    fn get_pair_frequencies(&self, word_splits: &Arc<HashMap<String, usize>>) -> HashMap<(String, String), usize> {
        
        // Parallelizes this process across the word splits
        word_splits.par_iter()
            .fold(

                // Each worker gets a new hash map!
                || HashMap::new(),
                
                |mut pairs, (word, &freq)| {
                
                    let symbols: Vec<_> = word.split_whitespace().collect();
                    
                    // Creates pairs from the split symbols
                    for i in 0..symbols.len() - 1 {
                
                        let pair = (symbols[i].to_string(), symbols[i + 1].to_string());
                        *pairs.entry(pair).or_insert(0) += freq;
                    }
                    
                    // Returns the generated pairs
                    pairs
                },
            )

            // Combines work from all workers!
            .reduce(
    
                || HashMap::new(),
                |mut a, b| {

                    // Add every entry of B in A; 
                    // 0 if new, else add the value :)

                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    
                    // Return the A HashMap!
                    a
                },
            )
    }

    fn merge_tokens(&self, pair: &(String, String), word_splits: &HashMap<String, usize>) -> HashMap<String, usize> {
        
        // Generating the pair pattern and replacement
        let pattern = format!("{} {}", &pair.0, &pair.1);
        let replacement = format!("{}{}", &pair.0, &pair.1);

        // Parallelizing the replacement process
        word_splits.par_iter()
            .map(|(word, &freq)| {

                // Replace all occurences of this pattern
                let new_word = word.replace(&pattern, &replacement);

                // Returns a new pair for the new token formed
                (new_word, freq)
            })
            .collect()
    }

    pub fn encode(&self, text: &str, max_length: Option<usize>) -> Vec<usize> {
        
        // Defining our encoded vector
        let mut encoded = Vec::new();
        let words = text.split_whitespace();
        
        for word in words {
            
            // Adding the word delimiter
            let word = format!("{}</w>", word);

            // Splitting into chars which will then be merged!
            let mut word = word.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
    
            // Applying merges
            loop {

                let mut changed = false;
                
                // Searching through all pairs
                for (pair, merged) in &self.merges {
                    
                    let pattern = format!("{} {}", pair.0, pair.1);
                    
                    // Found a match :)
                    if word.contains(&pattern) {
                        
                        word = word.replace(&pattern, merged);
                        changed = true;
                        break;
                    }
                }
                
                // Cannot merge anymore, so break!
                if !changed {
                    break;
                }
            }
    
            // Convert to token IDs
            for token in word.split_whitespace() {
                encoded.push(
                    self.vocab.get(token).copied().unwrap_or_else(|| self.vocab["<unknown>"])
                );
            }
        }
    
        // Creating final token sequence with BOS, EOS, and padding if not reached max length
        let mut final_tokens = vec![self.vocab["<begin_of_sentence>"]];
        
        if let Some(max_len) = max_length {

            // Truncating if it is needed
            if encoded.len() > max_len - 2 {  // -2 for BOS and EOS tokens :)
                encoded.truncate(max_len - 2);
            }

            final_tokens.extend(encoded);
            final_tokens.push(self.vocab["<end_of_sentence>"]);
            
            // Add padding tokens until we reach max_length
            while final_tokens.len() < max_len {
                final_tokens.push(self.vocab["<padding>"]);
            }
        }
        
        else {
            // If no max_length specified, just add tokens without padding
            final_tokens.extend(encoded);
            final_tokens.push(self.vocab["<end_of_sentence>"]);
        }
        
        final_tokens
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {

        // Flipping the vocabulary from string to tokens to tokens to strings!
        let reverse_vocab: HashMap<_, _> = self.vocab.iter()
            .map(|(k, &v)| (v, k))
            .collect();

        // Creating a text vector
        let mut text = Vec::new();
        
        for &token_id in token_ids {
            
            // Add a new pair if we find it except special tokens!
            if let Some(token) = reverse_vocab.get(&token_id) {
                
                // Skip special tokens :)
                if !self.special_tokens.contains(*token) {
                    text.push(token.to_string());
                }

            } 
            
            // Add unknown tokens otherwise
            else {
                text.push("<unknown>".to_string());
            }
        }

        // Finally, remove all occurences of the </w> token!
        text.join("")
            .replace("</w>", " ")
            .trim()
            .to_string()
    }

    pub fn save(&self, path: &str) -> Result<()> {

        // Creating a new JSON file in the location
        let file = std::fs::File::create(path)?;

        // Write our tokenizer to the file hehe
        serde_json::to_writer(file, self)?;

        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        
        // Open the created JSON file and load ourselves back in!
        let file = std::fs::File::open(path)?;
        let tokenizer = serde_json::from_reader(file)?;

        // OK ourselves :)
        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let mut tokenizer = BPETokenizer::new(1000);
        let texts = vec![
            "Hello world!".to_string(),
            "This is a test.".to_string(),
        ];
        
        tokenizer.train(&texts, 100).unwrap();
        
        let encoded = tokenizer.encode("Hello world!", None);
        let decoded = tokenizer.decode(&encoded);
        
        assert_eq!(decoded.trim(), "hello world!");
    }
}