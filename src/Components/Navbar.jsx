import { Link } from "react-router-dom"
import pink from '../Images/رؤؤر (1) 1.svg'
const Navbar = () => {
  return (
    <>
      <nav className="navbar navbar-expand-lg bg-body-tertiary shadow-sm ">
        <div className="container">
          <Link to='/' ><img className=" image img-fluid mb-2  " src={pink} alt="logo" /></Link>
          <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarSupportedContent">
            <ul className="navbar-nav me-auto mb-2 mb-lg-0">
              <li className="nav-item">
                <Link className="nav-link active ms-lg-3 home " aria-current="page" to='/' >Home</Link>
              </li>
              <li className="nav-item">
                <Link className="nav-link ms-lg-3 profile text-black-50  " to='/profile' >Profile</Link>
              </li>
            </ul>
            <form className="d-flex " role="search">
              <input className="form-control me-3 search p-2 " type="search" placeholder="Search" aria-label="Search" />
              <Link className="btn ms-2 me-2 signup" to='signup' >Signup</Link>
              <Link className="btn  login " to='login' >Login</Link>
            </form>
          </div>
        </div>
      </nav>
    </>
  )
}

export default Navbar