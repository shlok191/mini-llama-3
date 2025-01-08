import React, { useState } from 'react';

// Defining the needed variables that will be shared across the Prompt and GenerationBox!
interface RecordsContext {
  prompt: string;
  setPrompt: React.Dispatch<React.SetStateAction<string>>;
  generate: boolean;
  setGenerate: React.Dispatch<React.SetStateAction<boolean>>;

}

// Exporting the Records for future use :)
const Records = React.createContext<RecordsContext>({
  prompt: '',
  setPrompt: () => {},
  generate: false,
  setGenerate: () => {}
});

export const RecordsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  
  console.log("RecordsProvider rendering"); // Added to help debug

  const [prompt, setPrompt] = useState('');
  const [generate, setGenerate] = useState(false);

  return (
    <Records.Provider value={{
      prompt,
      setPrompt, 
      generate,
      setGenerate
    }}>
      {children}
    </Records.Provider>
  );
}

export default Records;