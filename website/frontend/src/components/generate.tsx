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
    const currentTextIndex = useRef(0); 
    const animationTimeout = useRef<number | null>(null);

    // Help with cancelling the streaming if need be!
    const abortControllerRef = useRef<AbortController | null>(null);
    const streamReaderRef = useRef<ReadableStreamDefaultReader<any> | null>(null);

    const { prompt, generate, temperature, top_k, setGenerate } = useContext(Records);

    // Function to cleanup ongoing stream and animations
    const cleanup = useCallback(() => {

        console.log("Beginning cleanup...")

        // Aborting any ongoing fetch request
        if (abortControllerRef.current) {

            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }

        // Releasing the reader if it exists
        if (streamReaderRef.current) {

            streamReaderRef.current.cancel();
            streamReaderRef.current = null;
        }

        // Clearing any pending animation timeouts
        if (animationTimeout.current) {
            clearTimeout(animationTimeout.current);
            animationTimeout.current = null;
        }

    }, []);
    
    const startGeneration = useCallback(async () => {
        console.log("Beginning generation...");
        
        // Create new abort controller
        abortControllerRef.current = new AbortController();
    
        setGeneratedText('');
        setError(null);
        currentTextIndex.current = 0;
    
        if(textContainerRef.current) {
            textContainerRef.current.innerHTML = '';
        }
        
        let fullText = prompt + ' ';
        setGeneratedText(fullText);
        
        const delay = async (ms: number) => new Promise(res => setTimeout(res, ms));
        await delay(300);

        try {
            const stream = await LLMService.generateText(
                model, 
                temperature, 
                top_k, 
                prompt,
                abortControllerRef.current.signal
            );
            
            const reader = stream.getReader();
            streamReaderRef.current = reader;

            const read = async () => {

                try {

                    // Check if we should stop reading
                    if (abortControllerRef.current?.signal.aborted) {
                        console.log("signal was cut!")
                        return;
                    }
    
                    const { done, value } = await reader.read();

                    if (done) {

                        console.log("Done was called!");
                        setGenerate(false);

                        if (onTextGenerated)
                            onTextGenerated(fullText);

                        streamReaderRef.current = null;
                        return;
                    }

                    fullText += value;
                    setGeneratedText(fullText);
                    
                    // Only continue reading if we haven't been aborted
                    if (!abortControllerRef.current?.signal.aborted) {
                        read();
                    }

                } catch (error) {
                    
                    if (error.name === 'AbortError') {
                        console.log('Stream was aborted');
                    } 
                    
                    else {
                        console.error('Stream error:', error);
                        setError(error instanceof Error ? error.message : String(error));
                    }

                    setGenerate(false);
                }
            };
    
            read();
            
        } catch (error) {

            console.error("An error occurred:", error);
            setGenerate(false);
        }
    }, [model, onTextGenerated, prompt, setGenerate, temperature, top_k]);

    useEffect(() => {

        if (prompt !== '' && generate === true) {

            console.log("Calling cleanup + startGeneration effect")

            startGeneration();
            return () => cleanup();
        }

    }, [generate, prompt, startGeneration, cleanup]);


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

        console.log("Calling the last useEffect!")
        
        if(textContainerRef.current)
        {
            setGenerate(false);
            textContainerRef.current.innerHTML = '';
        }

    }, [prompt, cleanup]);

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
