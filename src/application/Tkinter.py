
import hashlib
import ssl
import binascii
import os

global generator
global prime
global key_length
    
prime = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFCE0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B2699C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C93402849236C3FAB4D27C7026C1D4DCB2602646DEC9751E763DBA37BDF8FF9406AD9E530EE5DB382F413001AEB06A53ED9027D831179727B0865A8918DA3EDBEBCF9B14ED44CE6CBACED4BB1BDB7F1447E6CC254B332051512BD7AF426FB8F401378CD2BF5983CA01C64B92ECF032EA15D1721D03F482D7CE6E74FEF6D55E702F46980C82B5A84031900B1C9E59E7C97FBEC7E8F323A97A7E36CC88BE0F1D45B7FF585AC54BD407B22B4154AACC8F6D7EBF48E1D814CC5ED20F8037E0A79715EEF29BE32806A1D58BB7C5DA76F550AA3D8A1FBFF0EB19CCB1A313D55CDA56C9EC2EF29632387FE8D76E3C0468043E8F663F4860EE12BF2D5B0B7474D6E694F91E6DBE115974A3926F12FEE5E438777CB6A932DF8CD8BEC4D073B931BA3BC832B68D9DD300741FA7BF8AFC47ED2576F6936BA424663AAB639C5AE4F5683423B4742BF1C978238F16CBE39D652DE3FDB8BEFC848AD922222E04A4037C0713EB57A81A23F0C73473FC646CEA306B4BCBC8862F8385DDFA9D4B7FA2C087E879683303ED5BDD3A062B3CF5B3A278A66D2A13F83F44F82DDF310EE074AB6A364597E899A0255DC164F31CC50846851DF9AB48195DED7EA1B1D510BD7EE74D73FAF36BC31ECFA268359046F4EB879F924009438B481C6CD7889A002ED5EE382BC9190DA6FC026E479558E4475677E9AA9E3050E2765694DFC81F56E880B96E7160C980DD98EDD3DFFFFFFFFFFFFFFFFF
    #GLOBAL PRIMITIVE ROOT
generator = 2
key_length = 600
   
    '''
    --------------------------------------------------------
    ********* DIFFIE HELLMAN KEY EXCHANGE PROTOCOL *********
    --------------------------------------------------------
    '''
def generate_private_key(length):
		
	_rand = 0
	_bytes = length // 8 + 8
	while (_rand.bit_length() < length):
		hex_key = binascii.b2a_hex(os.urandom(_bytes))
		_rand = int(hex_key.encode('hex'),16)
	private_key = _rand
	return private_key
    
def generate_public_key(private_key):
	public_key = pow(generator, private_key, prime)
	return public_key

import Tkinter
from Tkinter import *
import webbrowser
import Tkinter, Tkconstants, tkFileDialog
import DH
import binascii
import os
import mysql.connector as mysql

def private_key_run():
    privatekeyBTN.config(state="disabled")
    DH.generate_private_key(10)
    print(textserve2)
    
def public_key_run():
    publickeyBTN.config(state="disabled")
    private_key = DH.generate_private_key(10)
    DH.generate_public_key(private_key)
    print(textserve1)
    
def filecreate_1():
    filename =  tkFileDialog.asksaveasfile(initialdir = "C:\Users\Diganta\Documents",title = "Select file",filetypes = (("text    files","*.txt"),("all files","*.*")))
    downloadfile1(filename)
    text1 = str(textserve2)
    filename.write(text1)
def filecreate_2():
    filename =  tkFileDialog.asksaveasfile(initialdir = "C:\Users\Diganta\Documents",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
    downloadfile2(filename)
    text2 = str(textserve1)
    filename.write(text2)
def downloadfile2(filename):
    textserve1.delete(0,END)
def downloadfile1(filename):
    textserve2.delete(0,END)
 
 global root
    root = Tkinter.Tk()
    root.wm_title("Create Private/Private key")
    blank = Entry(root)
    root.resizable(height=15, width=16)
    root.maxsize(500, 500)
    root.minsize(100, 150)
    Result1 = StringVar()
    Result2 = StringVar()
    
def quit():
    root.destroy()
quitBTN = Tkinter.Button(root, text='Exit', command = quit)
quitBTN.grid(row=4, column=7,sticky='E', padx=5, pady=2)
PublicKeyGet= LabelFrame(root, text='Public Key ')
PublicKeyGet.grid(row=0, columnspan=7, sticky='W', \
                   padx=9, pady=9, ipadx=9, ipady=9)
PrivateKeyGet = LabelFrame(root, text='Private Key ')
PrivateKeyGet.grid(row=0, column=7, sticky='W', \
                  padx=9, pady=9, ipadx=9, ipady =9) 
privatekeyBTN = Tkinter.Button(PrivateKeyGet, text='Create', command = private_key_run)
privatekeyBTN.grid(row=0, column=0, sticky='W', padx=5, pady=2)

downloadPriBTN = Tkinter.Button(PrivateKeyGet, text='Download', command = filecreate_1)
downloadPriBTN.grid(row=1, column=1, sticky='W', padx=5, pady=2)


textserve2 = Entry(PrivateKeyGet, textvariable=Result1)
textserve2.grid(row=0, column=1, sticky='W', padx=5, pady=2)
publickeyBTN = Tkinter.Button(PublicKeyGet, text='Create', command = public_key_run)
publickeyBTN.grid(row=2, column=0, sticky='W', padx=5, pady=2)

downloadPubBTN = Tkinter.Button(PublicKeyGet, text='Download', command = filecreate_2)
downloadPubBTN.grid(row=3, column=1, sticky='W', padx=5, pady=2)
   
textserve1 = Entry(PublicKeyGet, textvariable=Result2)
textserve1.grid(row=2, column=1, sticky='W', padx=5, pady=2)
  
Result1.set(privatekeyBTN)
Result2.set(publickeyBTN)
    
def connect():
    con = mysql.connect(host="localhost",user="root",password="Rootuser56#", database="keygen")
    cur = con.cursor()
    cur.execute("SELECT DATABASE()")
    data = cur.fetchone()
    print(\"Connection established to: \",data)
    con.commit()
    con.close()
def insert():
    con = mysql.connect(host=\"localhost\",user=\"root\",password=\"Rootuser56#\", database=\"keygen\")
    cur = con.cursor()
    sql = \"INSERT INTO users(public, private) VALUES(%d,%d)"
    public = input(textserve1)
    private = input(textserve2)
    val = (public,private)
    cur.execute(sql,val)
    con.commit()
SubmitBTN = Tkinter.Button(root, text='Connect', command = connect)
SubmitBTN.grid(row=4, column=1, sticky='E', padx=5, pady=2)
   
InsertBTN = Tkinter.Button(root, text='Insert', command = insert)
InsertBTN.grid(row=4, column=3, sticky='E', padx=5, pady=2)
    
root.mainloop()
   