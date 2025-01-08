import React, { useState } from 'react'
import Prompt from "../components/prompt.tsx"
import GenerativeBox from "../components/generate.tsx"
import { RecordsProvider } from '../contexts/records.tsx';

const WebPage: React.FC = () => {

    // Defining helpful state variables
    const [internalPrompt, setInternalPrompt] = useState<string>('');
    const [generatedText, setGeneratedText] = useState<string>('');
    const [isPirateTheme, setIsPirateTheme] = useState<boolean>(false);

    // This function will be passed to both Prompt and GenerativeBox!
    const handleGenerate = (promptText: string) => {
        setInternalPrompt(promptText);
    };

    // Callback to handle text generation completion
    const handleTextGenerated = (text: string) => {
        setGeneratedText(text);
    };

    // Function to toggle between themes
    const toggleTheme = () => {
        setIsPirateTheme(!isPirateTheme);
    };

  return (
    <RecordsProvider>
      <div className="page-container">
        <h1>{isPirateTheme ? "Ahoy, Matey! Spin a Yarn!" : "Welcome to the Story Weaver"}</h1>
        
        {/* Toggle for changing theme */}
        <button onClick={toggleTheme}>
          {isPirateTheme ? "Switch to Vanilla" : "Switch to Pirate"}
        </button>

        {/* Render the Prompt component */}
        <Prompt
          className={isPirateTheme ? 'pirate-theme' : 'vanilla-theme'}
        />

        {/* Render the GenerativeBox component */}
        <GenerativeBox 
          model={isPirateTheme ? 'blackbeard' : 'vanilla'}
          prompt={internalPrompt}
          temperature={0.75}
          top_k={8}
          className={isPirateTheme ? 'pirate-ui' : 'vanilla-ui'}
          onTextGenerated={handleTextGenerated}
          typingSpeed={10}
          fadeInDuration={50}
        />

        {/* Optionally show the last generated text */}
        {generatedText && (
          <div className="generated-text">
            <h3>Last Generated:</h3>
            <p>{generatedText}</p>
          </div>
        )}
      </div>
    </RecordsProvider>
  );
};

export default WebPage;