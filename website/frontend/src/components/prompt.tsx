import React, { useContext, ChangeEvent } from 'react';
import Records from '../contexts/records.tsx';
import { StyledPrompt, StyledInput, StyledButton } from './components.style.tsx';

const Prompt: React.FC = () => {

    // Defining a state for the prompt
    const { prompt, setPrompt, generate, setGenerate } = useContext(Records);
    
    const handleChange = (event: ChangeEvent<HTMLInputElement>) => {        
        setPrompt(event.target.value);
    };

    // Calls the `generate()` function defined in the Records context!
    const handleSubmit = () => {
        
        if (prompt.trim()) {

            // Do not do anything if we are currently generating!
            if(generate === false){
                setGenerate(true);
            }
        }
    };

    // Creating our DIV holding the prompt text box and the submission button
    return (
    
    <StyledPrompt>
        <StyledInput 
            type="text" 
            value={prompt}
            onChange={handleChange}
        />
        
        <StyledButton 
            onClick={handleSubmit}
        >
            Write!
        </StyledButton>
    </StyledPrompt>
    );
};

export default Prompt;