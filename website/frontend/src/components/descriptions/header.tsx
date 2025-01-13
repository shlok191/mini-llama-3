import React from 'react';

const Header: React.FC = () => {
    return (
       <div style = {{padding: '0px 50px 0px 50px', fontSize: '17px', fontFamily: 'var(--serious-font-family)' }}>
          <p>
            This project showcases a <b>50 Million parameter LLM</b> built and trained entirely from scratch on a single Nvidia A4000 GPU :)<br /><br />
            To apply my prior CUDA knowledge to LLMs, I wrote custom forward & backward CUDA kernels for a majority of the transformer architecture! <br />
            This included an efficient kernel for <b>Multi Headed Attention using Flash Attention 2 as reference and custom linear and embedding layers.</b>
            <br />I also wrote a <b>Rust Tokenizer</b> and used <b>PyTorch Lightning + W&B</b> to enhance training! <br /><br />

            I trained my model using the open sourced <b>Tiny Stories dataset</b> from Microsoft Research (and a pirate version of TinyStories with the arrr python package) for which there are links at the bottom!
          </p>
        </div>
    );
};

export default Header;