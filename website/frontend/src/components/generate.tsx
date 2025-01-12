import React, { useCallback, useState, useEffect, useRef, useContext } from 'react';
import LLMService from '../services/stream.tsx';
import Records from '../contexts/records.tsx';

// Defining an interface for creating our React container
interface GenerativeBoxProps {

    model: 'vanilla' | 'blackbeard';
    typingSpeed: number;
    fadeInDuration: number;
    onTextGenerated?: (text: string) => void;
}

const GenerativeBox: React.FC<GenerativeBoxProps> = ({
    model = 'vanilla',
    onTextGenerated,
    typingSpeed = 30,
    fadeInDuration = 20,
}) => {
    const [generatedText, setGeneratedText] = useState<string>('');
    const [error, setError] = useState<string | null>(null);

    const textContainerRef = useRef<HTMLDivElement>(null);
    const currentTextIndex = useRef(0); // Keep track of processed chars index
    const animationTimeout = useRef<number | null>(null);

    const { prompt, generate, temperature, top_k, setGenerate } = useContext(Records);

    const startGeneration = useCallback(async () => {


        console.log("Beginning generation...")

        // Starting from a clean state
        
        console.log("Beginning generation...")

        // Starting from a clean state
        setGeneratedText('');
        setError(null);
        currentTextIndex.current = 0;

        if(textContainerRef.current){
           textContainerRef.current.innerHTML = '';
        }
        
        let fullText = prompt + ' ';
        setGeneratedText(fullText);
        
        // Delaying to let the prompt be typed in!
        const delay = async (ms: number) => new Promise(res => setTimeout(res, ms));
        await delay(300);

        try {
            const stream = await LLMService.generateText(model, temperature, top_k, prompt);
            const reader = stream.getReader();

            const read = () => {
                
                reader.read().then(({ done, value }) => {
                    if (done) {
                        if (onTextGenerated) onTextGenerated(fullText);
                        return;
                    }

                    fullText += value;
                    setGeneratedText(fullText);
                    
                    if(generate)
                        read();
                });
            };

            read();
            setGenerate(false);
        }

        catch (error) {

            console.error(error);
            setError(error instanceof Error ? error.message : String(error));
            setGenerate(false);
        }
    }, [model, onTextGenerated, prompt, setGenerate, temperature, top_k, generate]);


    useEffect(() => {
        
        if (prompt !== '' && generate === true) {
            startGeneration();
        }
        
    }, [generate, prompt, startGeneration]);


    useEffect(() => {
    
        if (!textContainerRef.current) return;

        // Adding in tokens based on the text container reference
        const textContainer = textContainerRef.current;

        // Adding all the previous spans to the text container, so that the animation can take place
        let newText = generatedText.substring(currentTextIndex.current)
        
        // Breaking up the text for each character
        let newTextElements = newText.split('');

        // Going through the tokens and showing it up
        newTextElements.forEach((char, index) => {

            console.log("index: ", index)
            console.log("Current text index: ", currentTextIndex.current)
            console.log("Prompt length: ", prompt.length)

            // Creating the new span
            const span = document.createElement('span');
            span.textContent = char;
            span.style.opacity = '0'; // Initial opacity
            span.style.transition = `opacity ${fadeInDuration}ms ease-in`;
            span.style.fontSize = '20px';

            let temp_typingSpeed = typingSpeed;

            // Boldening the prompts
            if (currentTextIndex.current + index < prompt.length) { 
                span.style.fontWeight = '750';
                temp_typingSpeed *= 1.5;
            }

            textContainer.appendChild(span);

            // Setting a timeout to transition into full opacity
            animationTimeout.current = window.setTimeout(() => {
                span.style.opacity = '1';
            }, temp_typingSpeed * index);
        })
    
        currentTextIndex.current = generatedText.length;

        // This is critical to clear the timeout upon component unmounting.
        return () => {
            if (animationTimeout.current) {
                clearTimeout(animationTimeout.current);
            }
        };
        
    }, [generatedText, prompt, typingSpeed, fadeInDuration]);


    // Clear the container on new prompt
    useEffect(() => {
        return () => {
            if(textContainerRef.current)
            {
                setGenerate(false);
                textContainerRef.current.innerHTML = '';
            }
        }
    }, [prompt]);

    return (
        <div className={`relative`}>
            <div
                style={{
                    minHeight: '200px',
                    overflowY: 'auto',
                    padding: '1px 20px 1px 20px',
                }}
                ref={textContainerRef}
                className="whitespace-pre-wrap break-words"
                aria-live="polite"
                aria-busy={generate}
            />
            {error && (
                <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-md">
                    Error: {error}
                </div>
            )}
            {generate && (
                <div className="mt-2 text-sm text-gray-500">
                </div>
            )}
        </div>
    );
};

export default GenerativeBox;
