import { Link } from "react-router-dom"
import Apple from '../Images/Apple.svg'
import Face from '../Images/Facebook.svg'
import Google from '../Images/Google.svg'
const Login = () => {
  return (
    <>
      <div className="logPage d-flex flex-column  m-auto justify-content-center mt-5 ">
        <h1 className="fw-bold mb-3 ">Login To <span>Pink</span></h1>
        <p className="text-black-50 ">Welcome back! Please log in to access your account.</p>
      <form>
        <div className="mb-3">
          <label htmlFor="exampleInputEmail1" className="form-label">Email address</label>
          <input type="email" className="form-control" placeholder="Enter your Email" required id="exampleInputEmail1" aria-describedby="emailHelp"/>
          <div id="emailHelp" className="form-text">We will never share your email with anyone else.</div>
        </div>
        <div className="mb-3">
          <label htmlFor="exampleInputPassword1" className="form-label">Password</label>
          <input type="password" className="form-control" required placeholder="Enter your Password" id="exampleInputPassword1"/>
        </div>
        <div className="reforgot d-flex justify-content-between align-items-center ">
          <div className="mb-3 form-check">
            <label className="form-check-label" htmlFor="exampleCheck1">Remember me</label>
            <input type="checkbox" className="form-check-input" id="exampleCheck1"/>
          </div>
          <div className="mb-3 form-check">
          <Link to='forgot' className="forgot" >Forgot password?</Link>
        </div>
        </div>
        <button type="submit" className="btn btn-primary w-100  mb-2 fw-bold submit">Submit</button>
        <Link to='/signup' className="btn w-100 signup">Signup</Link>
    </form>
      <div className="con text-center mt-3">
        <p className="text-black-50 " >Or Continue with</p>
      </div>
      <div className="social d-flex justify-content-center align-items-center">
        <div className="google d-inline">
          <img src={Google} alt="Google" />
        </div>
        <div className="google d-inline mx-2 ">
          <img src={Face} alt="Facebook" />
        </div>
        <div className="google d-inline">
          <img src={Apple} alt="Apple" />
        </div>
      </div>
      </div>
    </>
  )
}

export default Login

