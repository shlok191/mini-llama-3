import React, { useContext, ChangeEvent } from 'react';
import Records from '../contexts/records.tsx'

interface PromptProps {
    className?: string;
}

const Prompt: React.FC<PromptProps> = ({ className = '' }) => {

    // Defining a state for the prompt
    const { prompt, setPrompt, generate } = useContext(Records);
    
    const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
        setPrompt(event.target.value);
    };

    // Calls the `generate()` function defined in the Records context!
    const handleSubmit = () => {
        
        if (prompt.trim()) {
            generate();
        }
    };

    // Creating our DIV holding the prompt text box and the submission button
    return (
    
    <div className={`${className}`}>
        <input 
            type="text" 
            value={prompt}
            onChange={handleChange}
            placeholder="What tale will you create?"
            className="prompt-input"
        />
        
        <button 
            onClick={handleSubmit}
            className="prompt-button"
        >
            Write!
        </button>
    </div>
    );
};

export default Prompt;