import React from 'react';

const Credits: React.FC = () => {
    return (
        <div style = {{padding: '0px 50px 0px 50px', fontFamily: 'var(--serious-font-family)'}}>
            <h2 style={{fontSize: '1.5rem', fontWeight: 'bold'}}>Credits</h2>
            <ul>
                <li>The TinyStories paper that showcases the dataset I used for this project: <a href="https://arxiv.org/pdf/2305.07759">TinyStories Paper</a></li>
                <li>A really helpful blog from Simon Boehm about highly efficient GEMM implementations: <a href="https://siboehm.com/articles/22/CUDA-MMM"> Simon's Blog </a></li>
                <li>The Flash attention paper that was really helpful for the attention implementation: <a href="https://arxiv.org/pdf/2307.08691">Flash Attention 2 Paper</a></li>
            </ul>
        </div>
    );
};

export default Credits;