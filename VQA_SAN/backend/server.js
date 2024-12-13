// const express = require('express');

// const cors = require('cors');

// const mongoose = require('mongoose');

// const jwt = require('jsonwebtoken');

// const cookieParser = require('cookie-parser');

// require('dotenv').config();

// const app = express();

// app.use(cors({ credentials: true, origin: 'http://localhost:3000' }));
// // app.use(cors({ credentials: true, origin: 'http://192.168.0.141:3000' }));
// // app.use(cors({ credentials: true, origin: 'http://192.168.187.187:3000' }));

// app.use(express.json());

// app.use(cookieParser());

// const dotenv = require('dotenv');

// const path = require('path');
// const envPath = path.resolve(__dirname,"./.env")
// console.log("envpath : ",envPath)


// dotenv.config({ path: envPath });

// console.log(process.env.MY_VARIABLE);

// mongoose
//   .connect(process.env.REACT_APP_MONGODB_URI, {
//     useNewUrlParser: true,
//     useUnifiedTopology: true,
//   })
//   .then(() => {
//     console.log('Connected to MongoDB');
//   })
//   .catch((err) => {
//     console.error('Failed to connect to MongoDB:', err);
//   });


// const userSchema = new mongoose.Schema({
//     username  : {type: String,required:true},
//     email : {type:String,required:true,unique:true},
//     password : {type:String,required:true}
// });

// // const signupitem = mongoose.model('signup',userSchema);
// const SignupItem = mongoose.model('SignupItem', userSchema);
// // console.log("SignupItem : ",SignupItem)
// app.post('/api/signin',(req,res) => {
//     console.log("req body : ",req.body)
//     console.log("enter into app.post of signin")
//     const newsignupitem = new SignupItem({username:req.body.username,email:req.body.email,password:req.body.password})

//     newsignupitem
//         .save()
//         .then((result) => {
//             console.log("result signup")
//             console.log("username,email,password",result);
//             res.sendStatus(200);
//         })
// });


// app.post('/api/login', async(req,res) => {
//   const { email , password } = req.body;
//   try {
//     const user = await SignupItem.findOne({ email});
//     if(!user){
//       localStorage.setItem('name','no user');
//       return res.status(401).json({error: 'Invalid email or password'});
//     }
//     if(user.password!==password){
//       return res.status(401).json({error:'wrong password'});
//     }
//     if(user.password===password){
//       localStorage.setItem('name',user);
//       res.status(200).json({Message:"Login success"});
//     }
//   }
//   catch (error) {
//     console.error('Login error:', error);
//     res.status(500).json({ error: 'An error occurred during login' });
//   }
// });


// const port = 5001;
// app.listen(port, () => {
//   console.log(`Server running on port ${port}`);
// });






// const express = require('express');
// const cors = require('cors');
// const mongoose = require('mongoose');
// const bcrypt = require('bcrypt');
// const jwt = require('jsonwebtoken');
// const cookieParser = require('cookie-parser');
// const dotenv = require('dotenv');
// require('dotenv').config();
// const path = require('path');

// const app = express();

// app.use(cors({ credentials: true, origin: 'http://localhost:3000' }));
// app.use(express.json());
// app.use(cookieParser());

// const envPath = path.resolve(__dirname, "./.env");
// dotenv.config({ path: envPath });

// mongoose
//   .connect(process.env.REACT_APP_MONGODB_URI, {
//     useNewUrlParser: true,
//     useUnifiedTopology: true,
//   })
//   .then(() => console.log('Connected to MongoDB'))
//   .catch((err) => console.error('Failed to connect to MongoDB:', err));

// const userSchema = new mongoose.Schema({
//   username: { type: String, required: true },
//   email: { type: String, required: true, unique: true },
//   password: { type: String, required: true },
// });

// const SignupItem = mongoose.model('SignupItem', userSchema);

// app.post('/api/signin', async (req, res) => {
//   try {
//     const { username, email, password } = req.body;
//     const hashedPassword = await bcrypt.hash(password, 10);
//     const newUser = new SignupItem({ username, email, password: hashedPassword });

//     await newUser.save();
//     res.sendStatus(200);
//   } catch (error) {
//     console.error('Signup error:', error);
//     res.status(500).json({ error: 'Error signing up user' });
//   }
// });

// app.post('/api/login', async (req, res) => {
//   const { email, password } = req.body;

//   try {
//     const user = await SignupItem.findOne({ email });
//     if (!user) return res.status(401).json({ error: 'Invalid email or password' });

//     const match = await bcrypt.compare(password, user.password);
//     if (!match) return res.status(401).json({ error: 'Invalid email or password' });

//     const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
//     res.cookie('auth_token', token, { httpOnly: true, secure: process.env.NODE_ENV === 'production' });

//     res.status(200).json({ message: 'Login successful', username:user.username});
//   } catch (error) {
//     console.error('Login error:', error);
//     res.status(500).json({ error: 'An error occurred during login' });
//   }
// });

// const port = 5001;
// app.listen(port, () => {
//   console.log(`Server running on port ${port}`);
// });







const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const cookieParser = require('cookie-parser');
const dotenv = require('dotenv');
const path = require('path');

// Initialize the app
const app = express();

// Enable CORS for frontend requests
app.use(cors({ credentials: true, origin: 'http://localhost:3000' }));
app.use(express.json()); // For parsing JSON bodies
app.use(cookieParser());

// Load environment variables
const envPath = path.resolve(__dirname, './.env');
dotenv.config({ path: envPath });

// Connect to MongoDB
mongoose
  .connect(process.env.REACT_APP_MONGODB_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => console.error('Failed to connect to MongoDB:', err));

// Define the User schema
const userSchema = new mongoose.Schema({
  username: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
});

// Model for users
const SignupItem = mongoose.model('SignupItem', userSchema);

// Sign-up route (POST)
app.post('/api/signin', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10); // Hash the password

    const newUser = new SignupItem({ username, email, password: hashedPassword });
    await newUser.save();
    res.sendStatus(200);
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ error: 'Error signing up user' });
  }
});

// Login route (POST)
app.post('/api/login', async (req, res) => {
  const { email, password } = req.body;

  try {
    const user = await SignupItem.findOne({ email });
    if (!user) return res.status(401).json({ error: 'Invalid email or password' });

    const match = await bcrypt.compare(password, user.password); // Compare hashed passwords
    if (!match) return res.status(401).json({ error: 'Invalid email or password' });

    // Create JWT token
    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
    res.cookie('auth_token', token, { httpOnly: true, secure: process.env.NODE_ENV === 'production' });

    // Respond with success and username
    res.status(200).json({ message: 'Login successful', username: user.username });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'An error occurred during login' });
  }
});

// Start the server
const port = 5001;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
