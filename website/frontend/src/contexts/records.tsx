import React, { useState } from 'react';

// Defining the needed variables that will be shared across the Prompt and GenerationBox!
interface RecordsContext {
  prompt: string;
  setPrompt: React.Dispatch<React.SetStateAction<string>>;
  generate: () => void;
}

// Exporting the Records for future use :)
const Records = React.createContext<RecordsContext>({
  prompt: '',
  setPrompt: () => {},
  generate: () => {},
});

export const RecordsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  
  console.log("RecordsProvider rendering"); // Added to help debug

  const [prompt, setPrompt] = useState('');

  const generate = () => {
    console.log('Generating with prompt:', prompt);
  }

  return (
    <Records.Provider value={{
      prompt,
      setPrompt, 
      generate
    }}>
      {children}
    </Records.Provider>
  );
}

export default Records;