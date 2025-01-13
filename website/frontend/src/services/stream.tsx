// Points to the Azure VM open port
const SERVER_URL = "https://blackbeard-shanty.com:2000";

const LLMService = {
  
    // Returns an asynchronous stream for a smoother experience :)
    generateText: async (model, temperature, top_k, prompt) => {

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
                })
            });
    
            // Checking if the response is null
            if (response.body === null) {
                throw new Error("Response body is null, cannot create reader.");
            }

            // Checking if the response is valid
            if(!response.ok){
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Now defining some readers and streams!
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            return new ReadableStream({
                async start(controller) {

                    while (true) {

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
                }
            });
        } 

        catch (error) {
            console.error("There was an error in the LLM Service! \n", error);
            throw error;
        }
    },
};

export default LLMService;
