var request = require('request-promise')

async function arraysum(){

    var data = {
        array: [1,3,1,0,0,42,39,34]
    }
    var options = {
        method: 'POST',
        uri: 'http://127.0.0.1:5000/predict',
        body: data,
        json: true
    }

    var sendRequest = await request(options)
    .then(function(parsebody){
        let result
        result = parsebody['result']
        console.log(result)
    }).catch(function(err){
        console.log(err)
    })
}

arraysum()