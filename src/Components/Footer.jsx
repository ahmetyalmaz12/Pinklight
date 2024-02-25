import smile from "../Images/smiley.svg"
import mention from "../Images/mention.svg"
import send from "../Images/paper-airplane.svg"
const Footer = () => {
  return (
    <>
    <div className=" footer container " >
      <img src={smile} className=" img-fluid" alt="smile" />
      <input type="text" className="form-control fs-5 text-black-50 " id="name" placeholder="Start typing..." />
      <img src={mention} className=" img-fluid me-2 mention " alt="mention" />
      <img src={send} className=" img-fluid" alt="send" />
    </div>
    </>
  )
}

export default Footer