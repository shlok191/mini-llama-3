import React from 'react';

const Links: React.FC = () => {
    return (
        <div style = {{padding: '0px 50px 0px 50px', fontFamily: 'var(--serious-font-family)'}}>
            <h2 style={{fontSize: '1.5rem', fontWeight: 'bold'}}>Links</h2>
            <ul>
                <li>Link to the GitHub repository: <a href="https://github.com/shlok191/mini-llama-3">Mini Llama 3 Project</a></li>
                <li>Link to attention implementation: <a href="https://github.com/shlok191/mini-llama-3/blob/main/model/cuda/attention/attention.cu">Attention.cu file</a></li>
                <li>Link to the Linear layer GEMM implementation: <a href="https://github.com/shlok191/mini-llama-3/blob/main/model/cuda/linear/linear.cu">Linear GEMM implementation</a></li>
                <li>Link to the Rust Tokenizer: <a href="https://github.com/shlok191/mini-llama-3/blob/main/model/src/tokenizers/rust_tokenizer/src/lib.rs">Rust tokenizer code</a></li>
            </ul>
        </div>
    );
};

export default Links;