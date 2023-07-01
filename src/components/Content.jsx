import React from 'react'
import Typewriter from 'typewriter-effect';

const Content = () => {
    return (
        <div className='text-white'>
            <div className='max-w-[800px] mt-[-96px] w-full h-screen text-center flex flex-col justify-center'>
                <p className='text-[#ffffff] font-bold p-2'>box 1</p>
                <h1 className='md:text-7xl sm:text-6xl text-4xl font-bold md:py-6'>
                <Typewriter
                    options={{
                        strings: ['Notes?', 'No problem.'],
                        autoStart: true,
                        loop: true,
                    }}
                />
                </h1>
                <div>
                    <p className='md:text-5xl sm:text-4xl text-xl font-bold'>lol</p>
                </div>
            </div>
        </div>
    )
}



export default Content