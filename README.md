Cloud security is one of the main concerns in the cloud computing domain. Sharing personal and sensitive information on a third-party medium poses serious risks of data eavesdropping and data misuse by any person with malicious intent. The traditional methods of secure file transfer and information are superfluous in the scenario of cloud. Extensive research and study is undergoing in this field to make cloud more secure and reliable. Some of the methods that stand out include AES encryption and Diffie Hellman Key Exchange. The latter method is so powerful that it may take millions of years for even the most powerful computers of current times to crack the code and reads the file. Our approach proposes a method that involves encrypting the file using any standard encryption technique and using Diffie Hellman for user authentication. In this way the files can be transferred in a public domain securely without the threat of being used by any unauthorized person.
Using Diffehelman Key exchange Key is generated
![](src/application/key%20generation.jpeg)
Diffie–Hellman key exchange establishes a shared secret between two parties that can be used for secret communication for exchanging data over a public network. The above conceptual diagram illustrates the general idea of the key exchange by using colors instead of very large numbers. The process begins by having the two parties, Alice and Bob, agree on an arbitrary starting color that does not need to be kept secret in this example the color is yellow. Each of them selects a secret color that they keep to themselves – in this case, orange and blue-green. The crucial part of the process is that Alice and Bob each mix their own secret color together with their mutually shared color, resulting in orange-tan and light-blue mixtures respectively, and then publicly exchange the two mixed colors. Finally, each of the two mixes the color he or she received from the partner with his or her own private color. The result is a final color mixture (yellow-brown in this case) that is identical to the partner's final color mixture. If a third party listened to the exchange, it would be computationally difficult for this party to determine the secret colors. In fact, when using large numbers rather than colors, this action is computationally expensive for modern supercomputers to do in a reasonable amount of time.
![]
Method
 Follow the mathematical implementation of Diffie Hellman key exchange protocol.
 p is a prime number. 
g is a primitive root modulo of p 
1. Alice and Bob agree to use a modulus p = 23 and base g = 5 
2. Alice gets her private key (key which she should not share with anyone) generated as 4. 
3. Thus, public key generated for Alice shall be 54%23 = 625%23 = 4
 4. Bob gets his private key (key which he should not share with anyone) generated as 3.
 5. Thus, public key generated for Bob shall be 53%23 = 125%23 = 10
 6. Now, Alice gets the public key of Bob and generates a secret key. i.e. (public key of Bob Private Key of Alice) mod p => (104 ) % 23 => 10000 % 23 => 18
 7. On the other side, Bob also uses a similar method to generate a secret key i.e. (public key of Alice Private Key of Bob) mod p => (43 ) % 23 => 64 % 23 => 18
 Encryption
 Encryption is widely used on the internet to protect user information being sent between a browser and a server, including passwords, payment information and other personal information that should be considered private. Organizations and individuals also commonly use encryption to protect sensitive data stored on computers, servers and mobile devices like phones or tablets. There are various encryption techniques that are present some of which are:
 • Triple DES
 • Blowfish 
• RSA 
• Two fish 
• AES 
The technique that we have used in our project is AES and it is described below.
![](src/application/gui.jpeg)
