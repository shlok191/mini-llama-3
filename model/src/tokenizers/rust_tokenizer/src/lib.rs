use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use pyo3::prelude::*;
use pyo3::types::PyType;
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

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[classmethod]
    fn load(_cls: &PyType, path: &str) -> PyResult<Self> {
        BPETokenizer::load(path)
            .map(|inner| PyTokenizer { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
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
    #[serde(serialize_with="serialize_merges", deserialize_with="deserialize_merges")]
    merges: HashMap<(String, String), String>,
    special_tokens: HashSet<String>,
    vocab_size: usize,
    iterations: usize,
}

// Custom serialization for merges
fn serialize_merges<S>(merges: &HashMap<(String, String), String>, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer{
    
    let serializable_merges: HashMap<String, String> = merges.iter()
        .map(|((first, second), merged)| {
    
            (format!("{}|{}", first, second), merged.clone())
        })
        .collect();
    
    serializable_merges.serialize(serializer)
}

// Custom deserialization for merges
fn deserialize_merges<'de, D>(deserializer: D) -> Result<HashMap<(String, String), String>, D::Error> where D: serde::Deserializer<'de>
{
    use serde::de::Error;
    
    let serializable_merges: HashMap<String, String> = 
        HashMap::deserialize(deserializer)?;
    
    let merges: HashMap<(String, String), String> = serializable_merges.into_iter()
        .map(|(key, value)| {
        
            let parts: Vec<&str> = key.split('|').collect();
        
            if parts.len() != 2 {
                return Err(D::Error::custom("Invalid merge key format"));
            }
        
            Ok(((parts[0].to_string(), parts[1].to_string()), value))
        })
        .collect::<Result<_, _>>()?;
    
    Ok(merges)
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
        let basic_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/'~\"\\";
        
        for c in basic_chars.chars() {
            
            // Add everything that is new
            if !tokenizer.vocab.contains_key(&c.to_string()) {
                tokenizer.vocab.insert(c.to_string(), tokenizer.vocab.len());
            }
        }

        println!("Finished initializing tokenizer with {} unique tokens!\n", tokenizer.vocab.len());
        
        // I like how Rust does not even have return statements, rad :)
        tokenizer
    }

    pub fn train(&mut self, texts: &[String]) -> Result<()> {
        
        println!("Starting training with {} texts\n", texts.len());
        
        // Phase 1: Building word frequencies
        println!("Building word frequencies...");
     
        // Defining the first progress bar for word frequency counting
        let word_freq_bar = ProgressBar::new(texts.len() as u64);
        
        word_freq_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} 🏴‍☠️ [{elapsed_precise}] [{eta_precise}] {bar:40.green/white} {pos:>7}/{len:7} {msg}")
                .unwrap()
        );
        
        // Keeps track of word frequencies
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        
        for text in texts {
            
            // Splits text into words and adds </w> to the end of each word for delimiting purposes!
            for word in text.split_whitespace() {

                let word = format!("{}</w>", word);
                *word_freqs.entry(word).or_insert(0) += 1;
            }

            word_freq_bar.inc(1);
            
            // Updating the message occasionally
            if word_freqs.len() % 1000 == 0 {
                word_freq_bar.set_message(format!("Unique words: {}", word_freqs.len()));
            }
        }
        
        word_freq_bar.finish_with_message(format!("Found {} unique words", word_freqs.len()));
        println!("\nConverting to character sequences...");
        
        // Phase 2: Merging the most adjacent occuring token pairs!

        // Convert words to character sequences
        let mut word_pieces: HashMap<String, usize> = word_freqs.iter()
            .map(|(word, &freq)| {
                let chars = word.chars()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                (chars, freq)
            })
            .collect();
     
        // Creating a progress bar!
        let progress_bar = ProgressBar::new(self.iterations as u64);
        
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} 🏴‍☠️ [{elapsed_precise}] [{eta_precise}] {bar:40.green/white} {pos:>7}/{len:7} {msg}")
                .unwrap()
        );
     

        println!("Starting main training loop...");
     
        // Defining our training loop
        for i in 0..self.iterations {
            
            // Quit if we have reached maximum possible tokens
            
            if self.vocab.len() >= self.vocab_size {
                progress_bar.finish_with_message("Reached maximum vocabulary size! Arrr!");
                break;
            }
     
            // Getting pair frequencies in parallel
            let pairs = self.get_pair_frequencies(&Arc::new(word_pieces.clone()));
            
            if pairs.is_empty() {
                progress_bar.finish_with_message("No more pairs to merge!");
                break;
            }
     
            // Finding the most frequently occurring token pair
            let best_pair = pairs.iter()
                .max_by_key(|&(_, count)| count)
                .ok_or_else(|| anyhow!("No pairs found"))?
                .0
                .clone();
     
            // Merging this best pair
            let merged = format!("{}{}", best_pair.0, best_pair.1);
            
            self.merges.insert(best_pair.clone(), merged.clone());
            self.vocab.insert(merged, self.vocab.len());
            
            // Merging the tokens wherever they occur!
            word_pieces = self.merge_tokens(&best_pair, &word_pieces);
            
            // Updating our progress bar 
            if (i + 1) % 1000 == 0 {
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
        
        // Defining patterns that should be merged in along with the replacement!
        let pattern = format!("{} {}", &pair.0, &pair.1);
        let replacement = format!("{}{}", &pair.0, &pair.1);
        
        // Parallelizing the replacement process across workers :)
        word_splits.par_iter()
            .map(|(word, &freq)| {
                let new_word = word.replace(&pattern, &replacement);
                (new_word, freq)
            })
            .collect()
    }

    pub fn encode(&self, text: &str, max_length: Option<usize>) -> Vec<usize> {

        // Stores our final encoded tokens
        let mut encoded = Vec::new();
        
        // Processing each word individually
        for word in text.split_whitespace() {

            let word = format!("{}</w>", word);
            
            // Splitting each word down to a single characters and then merging them back! :)
            let mut current = word.chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            
            
            'outer: loop {

                let mut changed = false;
                
                // All tokens of the current word
                let tokens: Vec<&str> = current.split_whitespace().collect();
                
                for i in 0..tokens.len() - 1 {
                    
                    // Forming all possible pairs of tokens
                    let pair = (tokens[i].to_string(), tokens[i + 1].to_string());

                    if let Some(merged) = self.merges.get(&pair) {

                        let pattern = format!("{} {}", pair.0, pair.1);
                        
                        if let Some(pos) = current.find(&pattern) {
                        
                            let before = &current[..pos];
                            let after = &current[pos + pattern.len()..];
                            
                            // Replacing the tokens!
                            current = format!("{}{}{}", before, merged, after);
                            changed = true;
                        
                            break;
                        }
                    }
                }
                
                if !changed {
                    break 'outer;
                }
            }
            
            for token in current.split_whitespace() {
                
                // Replacing unknown tokens with the unknown token ID
                let token_id = self.vocab.get(token).copied()
                    .unwrap_or_else(|| self.vocab["<unknown>"]);
                
                encoded.push(token_id);
            }
        }
        
        // Finally, we add the special tokens and padding tokens!
        let mut final_tokens = vec![self.vocab["<begin_of_sentence>"]];
        
        if let Some(max_len) = max_length {

            if encoded.len() > max_len - 2 {
                encoded.truncate(max_len - 2);
            }

            final_tokens.extend(encoded);
            final_tokens.push(self.vocab["<end_of_sentence>"]);
            
            while final_tokens.len() < max_len {
                final_tokens.push(self.vocab["<padding>"]);
            }
        } 
        
        else {
            final_tokens.extend(encoded);
            final_tokens.push(self.vocab["<end_of_sentence>"]);
        }
        
        // Return the final tokens :)
        final_tokens
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {

        // Flipping the vocabulary from string to tokens to tokens to strings!
        let reverse_vocab: HashMap<_, _> = self.vocab.iter()
            .map(|(k, &v)| (v, k))
            .collect();

        // Creating a text vector
        let mut text = Vec::new();
        
        let ignore_strs = vec!["<padding>", "<begin_of_sentence>", "<end_of_sentence>"];

        for &token_id in token_ids {
            
            // Add a new pair if we find it except special tokens!
            if let Some(token) = reverse_vocab.get(&token_id) {
                
                // Skip special tokens except the UNKNOWN string :)
                if !ignore_strs.contains(&token.as_str()) {
                    text.push(token.to_string());
                }
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
        let file = std::fs::File::create(path.to_string())?;

        // Write our tokenizer to the file hehe
        serde_json::to_writer(file, self)?;

        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        
        // Open the created JSON file and load ourselves back in!
        let file = std::fs::File::open(path.to_string())?;
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