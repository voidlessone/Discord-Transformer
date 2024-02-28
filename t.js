var  WebSocket   = require('ws');
require("./gpt3encoder.js")
var tf = require("@tensorflow/tfjs-node-gpu")
tf.setBackend("cpu") 
var ws = new WebSocket("wss://gateway.discord.gg/?v=6&encoding=json");
var { Client } = require('pg')
const fs = require('fs');
const client = new Client({
  user: 'DDDDDDDDDDDDD',
  host: 'localhost',
  database: 'discord',
  password: 'DDDDDDDDDDDDDD',
  //port: 3211,
});
(async () => {
	await client.connect();
	
	var token = "HERK"
var payload = {
   op: 2,
   d: {
      token: token,
      intents: 512,
      properties: {
         $os: "linux",
         $browser: "chrome",
         $device: "chrome",
      },
   },
};


const sequenceLength = 256;
const stretchLengthThreshold = 4;

  

var DATASET = ""
var interval = 0;

ws.addEventListener("open", function open(x) {
   ws.send(JSON.stringify(payload));
});

const embeddingLayer = tf.layers.embedding({ 
  inputDim: 256, 
  outputDim: 1, 
  inputLength: 256 
}); 


const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [256, 1]}));
model.add(tf.layers.dense({units: 256}));
model.add(tf.layers.flatten())
model.add(tf.layers.dense({units: 256}));

// or via your model:
model.compile({loss: 'meanSquaredError', optimizer: tf.train.adam(0.0002)});


var canTrain = true;
ws.addEventListener("message", async function incoming(data) {
   var x = data.data;
   var payload = JSON.parse(x);
   
   const { t, event, op, d } = payload;

   switch (op) {
      // OPCODE 10 GIVES the HEARTBEAT INTERVAL, SO YOU CAN KEEP THE CONNECTION ALIVE
      case 10:
         const { heartbeat_interval } = d;
         setInterval(() => {
            ws.send(JSON.stringify({ op: 2, d: null }));
         }, heartbeat_interval);

         break;
   }
   switch (t) {
      // IF MESSAGE IS CREATED, IT WILL console.log IN THE CONSOLE
      case "MESSAGE_CREATE":
      {
        if (!d.content || d.content.length == 0) break;
	var values = [	
		 d.author.id,
		 d.id,
		 d.content,
		 d.channel_id,
		 d.guild_id,
		]
		console.log(d.author.username + ": " + d.content);
		await client.query("SET datestyle = US, MDY; ")
		const res =  await client.query("INSERT INTO messages ( author, msg_id, content, channel, guild) VALUES ($1, $2, $3, $4, $5)", 
			
			values
		)
		var tx = tf.util.encodeString(d.content.padEnd(256, "\0"))
		var ax = tf.tensor([tx]);
		//ax = ax.expandDims(1)
		const output = embeddingLayer.apply(ax); 
		
		if (canTrain) {
			model.fit(output, ax, {epochs: 250}).then(() => canTrain =true);
			canTrain = false;
		}
		var p = model.predict(output);
		console.log(p.dataSync())
		await model.save("file://decoder-model")
		
		 
		tf.dispose(ax);
		tf.dispose(output);
      }
   }
});

})()
