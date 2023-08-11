import React, { useState } from 'react';
import Typewriter from 'typewriter-effect';
import '../styles.css'
import uploadIcon from './assets/UploadIcon.png'
import blobPic from './assets/BlobPaper.png'

const Content = () => {
    const [uploadedImage, setUploadedImage] = useState(null);

    const handleImageUpload = async () => {
        try {
        const fileInput = document.querySelector('input[type="file"]');
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const response = await fetch('http://localhost:3001/api/upload', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            setUploadedImage(data.filename);
        } else {
            throw new Error('Failed to upload image');
        }
        } catch (error) {
        console.error(error);
        }
    };

    return (
        <div className='text-white'>
            <div className='max-w-[800px] mt-[-96px] w-full h-screen text-center flex flex-col justify-center'>
                <h1 className='md:text-7xl sm:text-6xl text-4xl md:py-6' style={{ fontFamily: 'Jost, sans-serif', fontSize: '75px', fontWeight: '500' }}>
                <Typewriter
                    options={{
                        strings: ['Notes?', 'No problem.'],
                        autoStart: true,
                        loop: true,
                    }}
                />
                </h1>
                <div className='flex flex-col items-center'>
                <div>
                    <div>
                        <input type="file" />
                    </div>
                    <button onClick={handleImageUpload} style={{ backgroundColor: '#FFCC4D', padding: '4px 16px', borderRadius: '30px', color: 'rgba(30, 30, 30, 0.8)', fontFamily: 'Jost, sans-serif', fontSize: '30px', width: '230px', marginTop: '20px', marginLeft: '100px', marginBottom: '20px', display: 'flex', alignItems: 'center' }}>
                        <span> Upload Now</span>
                        <img src={uploadIcon} alt="Upload Icon" style={{ marginLeft: '10px', height: '30px', width: 'auto' }} />
                    </button>
                    {uploadedImage && <p>Picture uploaded successfully!</p>}
                </div>
                    {/* <div>
                        <input type="file" />
                    </div>
                    <button type='button' onClick={handleSelectFile} style={{ backgroundColor: '#FFCC4D', padding: '4px 16px', borderRadius: '30px', color: 'rgba(30, 30, 30, 0.8)', fontFamily: 'Jost, sans-serif', fontSize: '30px', width: '230px', marginTop: '20px', display: 'flex', alignItems: 'center' }}>
                        <span> Upload Now</span>
                        <img src={uploadIcon} alt="Upload Icon" style={{ marginLeft: '10px', height: '30px', width: 'auto' }} />
                    </button> */}
                    <div className="absolute right-0 top-5">
                        <img src={blobPic} alt="Blob Paper" style={{ width: '490px', height: 'auto' }} />
                    </div>
                    
                    </div>
                    
                </div>
                <br></br>
                <br></br>
                <br></br>
                <br></br>
                <div className='flex flex-col items-center p-8 border border-gray-300 rounded-lg'>
                <h2 className="text-2xl font-bold mb-4">How to Use</h2>
                <p className="text-center">
                    Simply choose a file from your device, upload it to NowNotes, and sit back as your handwritten content is immediately transcribed to a Google doc for easy editing and sharing access! Note: As the application is currently in its primary deployment stages, black and white images of plain, neat handwriting.
                </p>

                <h2 className="text-2xl font-bold my-4">Future Developments</h2>
                <p className="text-center">
                    Goals for the future of NowNotes include improving the prediction model by leveraging pretrained models. We hope to expand the service to modalities beyond Google Docs to eliminate any accessibility issues. Other goals include allowing users to upload a stream of images for efficient transcription, as well as expanding NowNotes to a mobile app platform.
                </p>

                <h2 className="text-2xl font-bold my-4">About</h2>
                <p className="text-center">
                    NowNotes was developed by a team of first-year Georgia Tech computer science students. Open source software was consulted, and the following GitHub was referenced to build the prediction model: https://github.com/Breta01/handwriting-ocr/tree/master
                </p>
            </div>
        </div>
    )
    }  

export default Content