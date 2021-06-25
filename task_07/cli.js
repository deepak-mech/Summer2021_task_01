function docker_cmd(){

var i = document.getElementById("in1").value;

var xhr = new XMLHttpRequest();

xhr.open("GET" , "http://192.168.43.140/cgi-bin/cli.py?x="+i,true);

xhr.send();

xhr.onload = function (){
	var output = xhr.responseText;
	document.getElementById("d1").innerHTML = output;
}

}
