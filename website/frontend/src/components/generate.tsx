import React, { useCallback, useState, useEffect, useRef, useContext } from 'react';
import LLMService from '../services/stream.tsx';
import Records from '../contexts/records.tsx';

// Defining an interface for creating our React container
interface GenerativeBoxProps {

    model: 'vanilla' | 'blackbeard';
    temperature: number;
    top_k: number;
    prompt: string;
    typingSpeed: number;
    fadeInDuration: number;
    className: string;

    onTextGenerated?: (text: string) => void;
}

// Hopefully this ends up looking pretty good :)
const GenerativeBox: React.FC<GenerativeBoxProps> = ({ 
    model = 'vanilla', 
    temperature = 0.75,
    top_k = 8,
    className = 'vanilla-ui',
    onTextGenerated,
    typingSpeed = 10,
    fadeInDuration = 50,
}) => {

    // Defining some stateful variables for dynamic prints :)
    const [generatedText, setGeneratedText] = useState<string>('');
    const [isGenerating, setIsGenerating] = useState<boolean>(false);
    const [shouldGenerate, setShouldGenerate] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const textContainerRef = useRef<HTMLDivElement>(null);
    const previousTextLength = useRef(0);

    // Fetching the prompt from the global context
    const { prompt, generate } = useContext(Records);

    const startGeneration = useCallback(async () => {

        // Starting from a clean state
        setIsGenerating(true);
        setGeneratedText('');
        setError(null);

        try{

            // Getting our beautiful stream of tokens!
            const stream = await LLMService.generateText(model, temperature, top_k, prompt);
            const reader = stream.getReader();
            const decoder = new TextDecoder();

            let fullText = prompt + ' ';

            function read() {
                
                // This is what actually is presenting the text in a streamed fashion!
                reader.read().then(({ done, value }) => {

                    if (done) {
                        
                        // onTextGenerated will likely show the time taken for generation
                        setIsGenerating(false);
                        if (onTextGenerated) onTextGenerated(fullText);
                        return;
                    }

                    const chunk = decoder.decode( value, { stream : true });
                    fullText += chunk;

                    setGeneratedText(prevText => prevText + chunk);

                    // Reload the function until we're done!
                    read();
                });
            }
            
            // Start reading :)
            read();

            // This means we can generate again after!
            setShouldGenerate(true);
        }

        catch (error) {

            // Logging the error and updating the error state
            console.error(error);
            
            setError(error instanceof Error ? error.message : String(error));
            setIsGenerating(false);
        }
    }, [model, temperature, top_k, prompt, onTextGenerated]);


    // We start generation once the prompt is generated
    useEffect (() => {

        if (shouldGenerate === false){
            
            // Beginning generation now!
            setShouldGenerate(true);
            startGeneration();
        }

    }, [generate, shouldGenerate, startGeneration]);

    // Defining an effect for having our text fade in and out!
    useEffect(() => {

        if (textContainerRef.current == null) return;
        
        const newTextLength = generatedText.length;
        const fragment = document.createDocumentFragment();

        for (let i = previousTextLength.current; i < newTextLength; i++) {
            
            // Creating a new span for the nex text
            const span = document.createElement('span');
            span.textContent = generatedText[i];

            // Adding visual features
            span.style.opacity = '0';
            span.style.transition = `opacity ${fadeInDuration}ms ease-in`;
            
            // Let the prompt be boldened to help differentiation :)
            if (i < prompt.length) {
                span.className = 'font-semibold text-gray-700';
            }
              
            fragment.appendChild(span);
        }   

        // Adding the new text to the container
        textContainerRef.current.appendChild(fragment);

        requestAnimationFrame(() => {
            
            if (textContainerRef.current == null){
                return;
            }
            
            const newSpans = Array.from(textContainerRef.current.querySelectorAll('span')) as HTMLSpanElement[];
            
            // Having the new spans be faded-in!
            for (let i = previousTextLength.current; i < newTextLength; i++) {

                setTimeout(() => {
                    if (newSpans[i]) {
                        newSpans[i].style.opacity = '1';
                    }
                }, typingSpeed * (i - previousTextLength.current));
            }
        });

        // Updating the length
        previousTextLength.current = newTextLength;
    }, [generatedText, prompt, fadeInDuration, typingSpeed]);

    useEffect(() => {
        
        return () => {
        
            if (textContainerRef.current) {
                textContainerRef.current.innerHTML = '';
            }

            previousTextLength.current = 0;
        };
    }, [prompt]);

    return (
        <div className={`relative ${className}`}>
            <div
                ref={textContainerRef}
                className="whitespace-pre-wrap break-words"
                aria-live="polite"
                aria-busy={isGenerating}
            />
          
            {error && (
                <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-md">
                Error: {error}
                </div>
            )}
          
            {isGenerating && (
                <div className="mt-2 text-sm text-gray-500">
                Generating...
                </div>
            )}
            </div>
        );
};

export default GenerativeBox;