import React, { useContext } from 'react';
import Records from '../contexts/records.tsx';
import { Pillar, Image } from './components.style.tsx';

const customSizes = {
    'pirate.svg': { width: '90px', height: '180px' },
    'barrel.svg': { width: '65px', height: '100px' },
    'crossbones.svg': { width: '80px', height: '80px' },
    'map.svg': { width: '135px', height: '135px' },
    'pirate-hat.svg': { width: '100px', height: '100px' },
    'parrot.svg': { width: '50px', height: '50px' },
    'ship.svg': { width: '100px', height: '100px' },
    'tequila.svg': { width: '60px', height: '90px' },
    'money-bag.svg': { width: '100px', height: '100px' },
    'teddy-bear.svg': { width: '115px', height: '115px' },
    'origami.svg': { width: '100px', height: '100px' },
    'table-tennis.svg': { width: '85px', height: '85px' },
    'controller.svg': { width: '125px', height: '125px' },
    'tennis-ball.svg': { width: '70px', height: '70px' },
    'toy-truck.svg': { width: '110px', height: '110px' },
    'slingshot.svg': { width: '80px', height: '80px' },
    'rubiks-cube.svg': { width: '100px', height: '100px' },
    'lego-man.svg': { width: '60px', height: '60px' }
}

export const LeftPillar = () => {

    const { theme, selectedImages } = useContext(Records);

    // Defining the positions for rows with one and two images!
    const positionsOneImagePirate = [
        { top: '3%', left: '30%' }, // Centered
        { top: '35%', left: '18%' }, // Centered
        { top: '70%', left: '20%' }, // Centered
    ];

    const positionsTwoImagesPirate = [
        
        // One on the left, and one on the right!
        { top: '20%', left: '7%' },
        { top: '22.5%', left: '52%' },

        { top: '55%', left: '3%' },
        { top: '52%', left: '60%' },
    
        { top: '87%', left: '8%' },
        { top: '85%', left: '50%' },
    ];

    const positionsOneImageVanilla = [
        { top: '5%', left: '20%' }, // Centered
        { top: '35%', left: '18%' }, // Centered
        { top: '67%', left: '23%' }, // Centered
    ];

    const positionsTwoImagesVanilla = [
        
        // One on the left, and one on the right!
        { top: '21%', left: '0%' },
        { top: '21%', left: '50%' },

        { top: '53%', left: '3%' },
        { top: '53.5%', left: '47%' },
    
        { top: '87%', left: '1%' },
        { top: '84.5%', left: '55%' },
    ];

    return (
        
        <Pillar>
            {selectedImages.map((img, index) => {

                // Determining if this is an odd or even index for the alternating rows of images
                const isSingleImageRow = index % 3 === 0;
                
                // Determining the position of the image based on the index
                const position = isSingleImageRow ? 
                
                (theme ? 
                    positionsOneImagePirate[Math.floor(index / 3)] : 
                    positionsOneImageVanilla[Math.floor(index / 3)])
                
                : (theme ? 
                    positionsTwoImagesPirate[index - 1 - Math.floor(index / 3)] : 
                    positionsTwoImagesVanilla[index - 1 - Math.floor(index / 3)]);

                return (
                    <Image 
                        key={img + index} 
                        src={`/assets/${theme ? 'pirate-ui' : 'vanilla-ui'}/${img}`} 
                        alt={img}
                        style={{
                            ...position,
                            ...(customSizes[img]),
                            objectFit: 'contain',
                            height: 'auto'
                        }}
                    />
                );
            })}
        </Pillar>
    );
};

export const RightPillar = () => {

    const { theme, selectedImages } = useContext(Records);

    // Defining the positions for rows with one and two images!
    const positionsOneImagePirate = [
        { top: '3%', left: '35%' }, // Centered
        { top: '35%', left: '23%' }, // Centered
        { top: '70%', left: '25%' }, // Centered
    ];

    const positionsTwoImagesPirate = [
        
        // One on the left, and one on the right!
        { top: '20%', left: '12%' },
        { top: '22.5%', left: '57%' },

        { top: '55%', left: '8%' },
        { top: '52%', left: '65%' },
    
        { top: '87%', left: '13%' },
        { top: '85%', left: '55%' },
    ];

    const positionsOneImageVanilla = [
        { top: '5%', left: '27%' }, // Centered
        { top: '36%', left: '28.5%' }, // Centered
        { top: '67%', left: '28%' }, // Centered
    ];

    const positionsTwoImagesVanilla = [
        
        // One on the left, and one on the right!
        { top: '21%', left: '5%' },
        { top: '21%', left: '55%' },

        { top: '53%', left: '8%' },
        { top: '53.5%', left: '52%' },
    
        { top: '87%', left: '6%' },
        { top: '84.5%', left: '60%' },
    ];

    return (
        
        <Pillar>
            {selectedImages.map((img, index) => {

                // Determining if this is an odd or even index for the alternating rows of images
                const isSingleImageRow = index % 3 === 0;
                
                // Determining the position of the image based on the index
                const position = isSingleImageRow ? 
                
                (theme ? 
                    positionsOneImagePirate[Math.floor(index / 3)] : 
                    positionsOneImageVanilla[Math.floor(index / 3)])
                
                : (theme ? 
                    positionsTwoImagesPirate[index - 1 - Math.floor(index / 3)] : 
                    positionsTwoImagesVanilla[index - 1 - Math.floor(index / 3)]);

                return (
                    <Image 
                        key={img + index} 
                        src={`/assets/${theme ? 'pirate-ui' : 'vanilla-ui'}/${img}`} 
                        alt={img}
                        style={{
                            ...position,
                            ...(customSizes[img]),
                            objectFit: 'contain',
                            height: 'auto'
                        }}
                    />
                );
            })}
        </Pillar>
    );
};
