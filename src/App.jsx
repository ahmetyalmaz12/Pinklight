import './App.css'
import Login from './Components/Login'
import Signup from './Components/Signup'
import { Route , Routes  } from 'react-router-dom'
import DefaultLayout from './Components/DefaultLayout'
import Profile from './Components/Profile'
import Home from './Components/Home'
function App() {
  return (
<Routes>
        <Route path="/" element={<DefaultLayout />} >
          <Route index element={<Home />} />
          <Route path="profile" element={<Profile />} />
        </Route>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
      </Routes>
  )
}

export default App
