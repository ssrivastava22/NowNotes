import React from 'react'

const Navbar = () => {
    return (
        <div className="flex items-center h-24 max-w-[1240px] mx-auto px-4 text-white">
            <h1 className='text-3xl font-bold text-[#00df9a]'>NOW NOTES</h1>
            <ul className='flex'>
                <li className='p-4 ml-4 whitespace-nowrap'>About</li>
                <li className='p-4 whitespace-nowrap'>How to Use</li>
                <li className='p-4 whitespace-nowrap'>Meet the Team</li>
            </ul>
        </div>
    )
}

export default Navbar