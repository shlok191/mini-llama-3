import React, { useEffect, useState } from 'react';

// Defining the names of the custom images we have for pillar decoration!
const pirateImages = [
  'pirate.svg', 'barrel.svg', 'crossbones.svg', 'map.svg', 'pirate-hat.svg', 
  'parrot.svg', 'ship.svg', 'tequila.svg', 'money-bag.svg'
];

const vanillaImages = [
  'controller.svg', 'origami.svg', 'table-tennis.svg', 'teddy-bear.svg', 
  'tennis-ball.svg', 'toy-truck.svg', 'rubiks-cube.svg', 'slingshot.svg', 'lego-man.svg'
];


// Defining the needed variables that will be shared across the Prompt and GenerationBox!
interface RecordsContext {

  prompt: string;
  generate: boolean;
  temperature: number;
  top_k: number;
  theme: boolean;
  selectedImages: string[];
  music: HTMLAudioElement | null;

  setSelectedImages: React.Dispatch<React.SetStateAction<string[]>>;
  setPrompt: React.Dispatch<React.SetStateAction<string>>;
  setGenerate: React.Dispatch<React.SetStateAction<boolean>>;
  setTemperature: React.Dispatch<React.SetStateAction<number>>;
  setTopK: React.Dispatch<React.SetStateAction<number>>;
  setTheme: React.Dispatch<React.SetStateAction<boolean>>;
  setMusic: React.Dispatch<React.SetStateAction<HTMLAudioElement | null>>;

}

// Exporting the Records for future use :)
const Records = React.createContext<RecordsContext>({

  prompt: '',
  generate: false,
  temperature: 0.75,
  top_k: 8,
  theme: true,
  selectedImages: [],
  music: null,

  setPrompt: () => {},
  setGenerate: () => {},
  setTemperature: () => {},
  setTopK: () => {},
  setTheme: () => {},
  setSelectedImages: () => {},
  setMusic: () => {}
});

// Defining a provider which actually implements the functionality
export const RecordsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  
  const [prompt, setPrompt] = useState('');
  const [generate, setGenerate] = useState(false);
  const [temperature, setTemperature] = useState(0.75);
  const [top_k, setTopK] = useState(32);
  const [theme, setTheme] = useState(true);
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  const [music, setMusic] = useState<HTMLAudioElement | null>(null);

  // Defining a useEffect to populate the selected images
  useEffect(() => {
    setSelectedImages(theme ? pirateImages : vanillaImages);
  
  }, [theme]);

  return (
    <Records.Provider value={{
      prompt,
      generate,
      temperature,
      top_k,
      theme,
      selectedImages,
      music,

      setPrompt, 
      setGenerate,
      setTemperature,
      setTopK,
      setTheme,
      setSelectedImages,
      setMusic
    }}>
      {children}
    </Records.Provider>
  );
}

export default Records;