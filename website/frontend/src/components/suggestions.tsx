import React, { useContext } from 'react';
import Records from '../contexts/records.tsx'

import { StyledSuggestionContainer, StyledSuggestionBubble } from './components.style.tsx';

const SuggestionBubbles: React.FC = () => {
    
    const { theme, setPrompt } = useContext(Records);

    // Defining some suggestions for the user! :)
    const suggestions = theme ? 
    [
        "Once upon a time, in the",
        "The old house in the forest was ",
        "The pirate was playing with his mates, ",
    ] : 
    [
        "Once upon a time, there was ",
        "Inside the big forest, the ",
        "Jake was playing with his friends, "
    ];

    // Defining a function to update the prompt bubbles
    const handleSuggestionClick = (suggestion: string) => {
        setPrompt(suggestion);
    };
    
    // Returning the suggestions bubbles centered above the prompt box!
    return (
        <StyledSuggestionContainer>
            {suggestions.map((suggestion, index) => (
                <StyledSuggestionBubble
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion)}
                >
                    {suggestion}
                </StyledSuggestionBubble>
            ))}
        </StyledSuggestionContainer>
    );
};

export default SuggestionBubbles;