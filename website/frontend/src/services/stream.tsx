// Points to the Azure VM open port
const SERVER_URL = "https://blackbeard-shanty.com:2000";

const LLMService = {
    
    // Returns an asynchronous stream for a smoother experience :)
    generateText: async (model: string, temperature: number, top_k: number, prompt: string, signal?: AbortSignal) => {
        
        try {
            console.debug('Sending request with:', { model, temperature, top_k, prompt });

            const response = await fetch(`${SERVER_URL}/generation`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model, // The model name --> Vanilla or Blackbeard
                    temperature, // The temperature for generation --> Determines creativity + randomness
                    top_k, // The number of tokens to sample --> Also assists with creativity + randomness
                    prompt, // The prompt --> self explanatory I hope :) 
                }),
                signal // Add the abort signal to the fetch request
            });
    
            // Checking if the response is null
            if (response.body === null) {
                throw new Error("Response body is null, cannot create reader.");
            }

            // Checking if the response is valid
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Now defining some readers and streams!
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            return new ReadableStream({
                async start(controller) {
           
                    // Adding signal handler for stream cancellation
                    if (signal) {
                        signal.addEventListener('abort', () => {
                            reader.cancel();
                            controller.close();
                        });
                    }

                    try {
                        while (true) {

                            // Checking if aborted before each read
                            if (signal?.aborted) {
                                controller.close();
                                break;
                            }

                            // Keep reading as long as we can
                            const { done, value } = await reader.read();
                            
                            if (done) {
                                controller.close();
                                break;
                            }
                            
                            try {
                                controller.enqueue(decoder.decode(value, { stream: true }));
                            } catch (decodeError) {
                                console.error('Decoding error:', decodeError, 'Value:', Array.from(value));
                            }
                        }
                    } catch (error) {

                        // If the error is an AbortError, we don't want to propagate it as an error
                        if (error.name === 'AbortError') {
                            controller.close();
                        } else {
                            controller.error(error);
                        }
                    }
                },
                cancel() {
                    reader.cancel();
                }
            });
        } catch (error) {

            console.log("Got an error!")
            // Check if the error is due to abortion
            if (error.name === 'AbortError') {
                console.log('Request was aborted');
                throw error;
            }
            console.error("There was an error in the LLM Service! \n", error);
            throw error;
        }
    },
};

export default LLMService;