import React from 'react'
import logo from './assets/Logo.png'
import '../styles.css'


const Navbar = () => {
    return (
        <div className="flex items-center h-24 max-w-[1240px] mx-auto px-4 text-white">
            <h1 className='logo'> <img src={logo} alt='Logo' style={{ width: '200px', height: 'auto' }}></img> </h1>
            {/* <ul className='flex' style={{ fontFamily: 'Jost, sans-serif', fontSize: '20px', fontWeight: '500' }}>
                <li className='p-4 ml-4 whitespace-nowrap'>About</li>
                <li className='p-4 whitespace-nowrap'>How to Use</li>
                <li className='p-4 whitespace-nowrap'>Meet the Team</li>
            </ul> */}
        </div>
    )
}

export default Navbar