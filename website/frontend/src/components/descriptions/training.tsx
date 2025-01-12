import React from 'react';

const TrainingDetails: React.FC = () => {
    return (
        <div style={{ padding: '0px 50px 0px 50px', fontFamily: 'var(--serious-font-family)'}} id="training-details">
            <p>
                I trained my model using <b>PyTorch Lightning</b> for LR scheduling and checkpointing. I also used <b>Weights & Biases</b> for monitoring!
            </p>
             <p> Here are the hyperparameters that were used: </p>
             <ul style={{ paddingLeft: '20px' }}>
                <li><b>Batch Size:</b> 64</li>
                <li><b>Embedding Dimension:</b> 1024</li>
                <li><b>Number of Decoder Layers:</b> 4</li>
                <li><b>Number of Attention Heads:</b> 4</li>
                <li><b>Dropout:</b> 0.1</li>
                <li><b>Max Training Steps:</b> 100,000</li>
                <li><b>Learning Rate:</b> 1.5e-4</li>
                <li><b>Warmup Steps:</b> 1000</li>
                <li><b>LR Scheduler: </b> Cosine Annealing</li>
            </ul>
           <div style={{display: 'flex'}}>
                <div style={{marginLeft: '40px'}}>

                    <h3 style={{paddingLeft: "75px"}}>TinyStories Dataset Training</h3>
                    <img src="/assets/vanilla-validation-loss.png" alt = "Weights & Biases Graph" width='350px' height='250px'/>
                </div>
                
                 <div style={{marginLeft: '120px'}}>
                
                     <h3 style={{paddingLeft: "90px"}}>Pirate Dataset Finetuning</h3>
                     <img src="/assets/pirate-validation-loss.png" alt = "Weights & Biases Graph" width='350px' height='250px'/>
                
                 </div>
             </div>
             <p>
                During training, it was very exciting to watch the model learning language structure, punctuation and other details as time went on :)
            </p>
        </div>
    );
};

export default TrainingDetails;