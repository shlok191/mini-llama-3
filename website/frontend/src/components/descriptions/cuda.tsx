import React from 'react';

const CurrentDetails: React.FC = () => {
    return (
      <div style={{ padding: '0px 40px 0px 40px', fontFamily: 'var(--serious-font-family)' }} id="cuda-details">
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ width: '100%', paddingRight: '10px' }}>
            <h3 style={{fontSize: '1.5rem', fontWeight: 'bold'}}>Current Work</h3>
            <ul style={{ paddingLeft: '20px', fontSize: '18px'}}>
              <li style={{ marginTop: '1rem' }}>Created a batched Flash Attention 2 kernel in CUDA supporting MHA Transformers! <b>Here are the Technical Details:</b>
                <ul style={{ paddingLeft: '20px'}}>
                  <li style={{ marginTop: '1rem' }}>Utilized vectorized memory loads to fetch <b>4 floats per memory transaction</b> to reduce memory transfer bottleneck</li>
                  <li style={{ marginTop: '0.25rem' }}>Efficiently used <b>96 KB of available 100 KB</b> of shared memory per SM by tiling <b>32 Q rows</b> and <b>16 K, V columns</b> per kernel</li>
                  <li style={{ marginTop: '0.25rem' }}>Carefully analyzed kernel structure to deduce where the <b>cheaper syncwarp()</b> can replace the costlier syncthreads() calls!</li>
                  <li style={{ marginTop: '0.25rem' }}>Prioritized cheaper <b>warp level arithmetics</b> over <b>atomic operations</b> for inter-thread operations (particularly for online softmax)</li>
                  <li style={{ marginTop: '1rem' }}>Speedup over Base PyTorch Implementation (Forward Pass): <b>3.62 x</b></li>
                  <li style={{ marginTop: '0.25rem' }}>Speedup over Base PyTorch Implementation (Backward Pass): <b>4.78 x</b></li>
                </ul>
              </li>
                
              <li style={{ marginTop: '2rem' }}>Defined a batched Linear layer utilizing an efficient GEMM implementation achieving 90% of CuBLAS speed on average!</li>
              <li style={{ marginTop: '0.2rem' }}>Defined a custom Embedding layer maximizing work per thread, and tied it to the LM head to save memory</li>
                      
              <li style={{ marginTop: '0.2rem' }}> Lastly, for all custom implementations: </li>
              <ul style={{ paddingLeft: '20px' }}>
                <li style={{ marginTop: '1rem' }}>Utilized <b>CUDA streams</b> to allow for faster asynchronous processing across attention heads and batches</li>
                <li style={{ marginTop: '0.25rem' }}>Wrapped CUDA code using <b>Pybind 11</b> to expose to the <b>PyTorch autograd library!</b></li>
              </ul>
                
              <li style={{ marginTop: '1rem' }}>I also wrote a <b>Rust Tokenizer</b> trained using <b>Byte-Pair Encoding</b> to familiarize myself with the language :)</li>
            </ul>
          </div>
        </div>
        
        <div style={{ width: '100%'}}>
          <h3 style={{fontSize: '1.5rem', fontWeight: 'bold', marginTop: '1rem'}}>Future Work</h3>
            <ul style = {{ paddingLeft: '20px', fontSize: '18px'}}>
              <li style={{ marginTop: '1rem'}}>Adding support for Key Value (KV) caching to improve inference speeds along with other transformer optimizations</li>
              <li style={{ marginTop: '0.25rem'}}>Implementing Brain Float 16 (BFloat16) support to improve speeds of training!</li>
            </ul>
        </div>
    </div>
    );
};

export default CurrentDetails;