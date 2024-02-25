import Navbar from './Navbar'
import Footer from './Footer'
import { Outlet } from 'react-router-dom'
const DefaultLayout = () => {
  return (
    <>
      <div className="homeCon d-flex flex-column justify-content-between m-auto ">
        <Navbar />
        <div className="container">
        <Outlet />
        </div>
        <div className="foot mx-auto ">
        <Footer />
        </div>
      </div>
    </>
  )
}

export default DefaultLayout
