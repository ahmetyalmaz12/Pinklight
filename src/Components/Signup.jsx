import { Link } from "react-router-dom"
import Apple from '../Images/Apple.svg'
import Face from '../Images/Facebook.svg'
import Google from '../Images/Google.svg'
import { useState } from "react"
const Signup = () => {
  const [email, setEmail] = useState('');
  const [confirmEmail, setConfirmEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');

  const handleEmailChange = (e) => {
      setEmail(e.target.value);
      setError('');
  };

  const handleConfirmEmailChange = (e) => {
      setConfirmEmail(e.target.value);
      setError('');
  };
  const handlePasswordChange = (e) => {
    setPassword(e.target.value);
      setError('');
  };

  const handleConfirmPasswordChange = (e) => {
    setConfirmPassword(e.target.value);
      setError('');
  };
    const isSignUpEnabled = email !== '' && email === confirmEmail && password !== '' && password==confirmPassword ;
  return (
    <>
      <div className="logPage d-flex flex-column  m-auto justify-content-center mt-5 ">
        <h1 className="fw-bold mb-3 ">Create Account To <span>Pink</span></h1>
        <p className="text-black-50 ">Join To Our Commity Today </p>
      <form >
      <label htmlFor="name" className="form-label">Filll Name</label>
            <input type="text" className="form-control" id="name" placeholder="Enter you Fill Name" />
        <div className="email row">
          <div className="mb-3 col-md-6 ">
            <label htmlFor="exampleInputEmail1" className="form-label">Email</label>
            <input type="email" className="form-control" id="exampleInputEmail1" value={email} onChange={handleEmailChange} required placeholder="Enter your Email" aria-describedby="emailHelp"/>
          </div>
          <div className="mb-3 col-md-6 ">
            <label htmlFor="exampleInputEmail2" className="form-label">Confirm Email</label>
            <input type="email" className="form-control" id="exampleInputEmail2" value={confirmEmail} onChange={handleConfirmEmailChange} required placeholder="Enter your Confirm Email" aria-describedby="emailHelp"/>
          </div>
        </div>
        <div className="password row">
          <div className="mb-3 col-md-6 ">
            <label htmlFor="exampleInputPassword1" className="form-label">Password</label>
            <input type="password" className="form-control" value={password} onChange={handlePasswordChange} required placeholder="Enter your Password" id="exampleInputPassword1"/>
          </div>
          <div className="mb-3 col-md-6">
            <label htmlFor="exampleInputPassword2" className="form-label"> Confirm Password</label>
            <input type="password" className="form-control" value={confirmPassword} onChange={handleConfirmPasswordChange} required placeholder="Enter your Confirm Password" id="exampleInputPassword2"/>
          </div>
        </div>
        {error && <div className="alert alert-danger">{error}</div>}
        <button type="submit" disabled={!isSignUpEnabled} className="btn btn-primary w-100  mb-2 fw-bold submit">Create Account</button>
        <Link to='/login' className="btn w-100 signup">Login</Link>
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

export default Signup

