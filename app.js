var express = require('express'); 
var app = express(); 
const ejs=require("ejs");
const bodyparser=require("body-parser");
const path=require("path");
const {PythonShell} = require("python-shell");


app.use(express.static("public"));
app.use(bodyparser.urlencoded({extended:true}));


app.set('views',path.join(__dirname,'views'));
app.set('view engine','ejs');

  
// Creates a server which runs on port 3000 and  
// can be accessed through localhost:3000 
app.get('/output',function(req,res){
    res.render("output",{val:"ok"});
})
app.get('/',function(req,res){
    res.render("home",{val:"no"});
})

app.get('/foo',function(req,res){
    

    var spawn = require("child_process").spawn;
    const process = spawn('python', ['./webcam.py'])
   
    process.stdout.on('data', function(data) {
        // Do whatever you want with the returned data.
        // ...
       console.log(data);
       
    })

    // var pyshell = new PythonShell('./webcam.py');
    // pyshell.on('message', function (message) {
    //     // received a message sent from the Python script (a simple "print" statement)
    //     console.log(message);
    // });
    
   
    process.stderr.on('data', (data) =>
    {
        console.log(data);
    });
    process.stdout.on('data', (data) => {
        console.log(String.fromCharCode.apply(null, data));
        res.render("output",{val:data});
       
    });
    
    process.on('close', function(d) 
    {
        console.log("Closed"+d);
        
       // res.render("output");
    });
    
   
    
})





app.listen(3000, function() { 
    console.log('server running on port 3000'); 
} ) 
  













app.get('/fordebug', callName); 
  
function callName(req, res) { 
      
    
    console.log("uj");
    var spawn = require("child_process").spawn; 
   
     
   var process = spawn('python',['./hello.py'] ); 
  
   
    process.stdout.on('data', function(data) { 

        res.send(data.toString());
          
    } ) 
} 
  
app.get('/hi',function(req,res)
{
    res.redirect(__dirname + "hi.html");
})