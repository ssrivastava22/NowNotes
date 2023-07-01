import React from 'react'
import Typewriter from 'typewriter-effect';
import '../styles.css'
import uploadIcon from './assets/UploadIcon.png'
import blobPic from './assets/BlobPaper.png'

const Content = () => {
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
                    <button type='button' style={{ backgroundColor: '#FFCC4D', padding: '4px 16px', borderRadius: '30px', color: 'rgba(30, 30, 30, 0.8)', fontFamily: 'Jost, sans-serif', fontSize: '30px', width: '230px', marginTop: '20px', display: 'flex', alignItems: 'center' }}>
                        <span> Upload Now</span>
                        <img src={uploadIcon} alt="Upload Icon" style={{ marginLeft: '10px', height: '30px', width: 'auto' }} />
                    </button>
                    <div className="absolute right-0 top-5">
                        <img src={blobPic} alt="Blob Paper" style={{ width: '490px', height: 'auto' }} />
                    </div>
                </div>
            </div>
        </div>
    )
}



export default Content