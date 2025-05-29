Claude.md

Rules to follow:

This file provides guidance to Claude Code when working with code in this repository.

Create a new Branch at the beginning of this session.

Rules to Follow:

ALWAYS write extremly lean Python code. the goal is to fast doing research not production ready applications.
Always try to write as lean as possible code. Don't blow up the repo. 
MOVE Test scripts to the tests folder if they are not already there and ensure that they could be reused for later Tests. 
Test all finished issues and the end to ensure that the overall research goal is reached.
ALWAYS commit after each new function is added to our codebase
Ensure that you are using uv for isolating environments and packagemanagement
Use tree command for project structure.
Ensure that if you are finished with all issue a pull requests are created.
Create a tmp folder for development. And create a scratchpad.md file in this folder to chronologically document the development process.
Give the user after each finished step a short advise how to test your implementation. Write this in a test_advice.md file in the /tmp/ folder.
Write an overall documentation about how to use the research script and how to extend it easily.
Absolut important keep the repo lean and clean, don't add unnecessary files, don't overengineer.

Use the Following Services:
Ollama: Base URL: host.docker.internal:11434
