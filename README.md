# HeisenbergFormNLAProject

Motivated by recent developments in serverless systems for large-scale computation, we are going to implement OverSketched Newton proposed by Gupta et al. (2020), a randomized Hessian-based optimization algorithm to solve large-scale convex optimization problems in serverless systems. OverSketched Newton leverages matrix sketching ideas from Randomized Numerical Linear Algebra to compute the Hessian approximately. These sketching methods lead to inbuilt resiliency against stragglers that are a characteristic of serverless architectures. We will empirically validate convergence of the algorithm by solving large-scale supervised learning problems on real-world datasets using serverless platforms.

## Instruction how to run Jupyter Server on AWS EC2
1. Log into the AWS management console - for this you have to make an account in AWS, verify phone number, card number and email.

1. Search EC2 in the searchbar and click on it

<img width="660" alt="image" src="https://user-images.githubusercontent.com/82328870/208980933-0c74a69e-e204-4393-a687-6b14f2c63f28.png">

3. In the left menu click on Instances and click on "Launch instances"

<img width="1279" alt="image" src="https://user-images.githubusercontent.com/82328870/208981661-c77d97aa-1a0c-458d-b0c8-ac3ccfe15e15.png">

4. In instance setting you can select all options with "Fier free eligible" tag - this will be enough for our project. You can select other options, but this will require additional costs. In the screen below you can see the options, selected by us.

<img width="276" alt="image" src="https://user-images.githubusercontent.com/82328870/208982299-6d4303f4-0756-4c4f-b6d4-361ac0d2d0ff.png">

5. While selecting the key pair `ppk` file will be generated. Moreover, you should save public DNS from connection settings of your instance

<img width="593" alt="image" src="https://user-images.githubusercontent.com/82328870/208983109-e64e48c8-d785-4e21-8520-968ee25d69b7.png">

6. Download PuTTY app from https://www.puttygen.com/ In the host name put [username]@[public DNS]

<img width="334" alt="image" src="https://user-images.githubusercontent.com/82328870/208983782-e8fa294b-af3d-4137-9c1a-155d3552f00d.png">

7. In Connection - SSH - Tunnels put 8888 as port and http://127.0.0.1:8888 as Destination - usual setting for your local Jupyter session.

<img width="333" alt="image" src="https://user-images.githubusercontent.com/82328870/208984023-30c7de3e-ce15-4258-b790-db5ae39f6aee.png">

8. In Connection - SSH - Auth -credentials click "Browse" near the "Public-key file for authentication" space and put the destination to the previously generated `ppk` file. And click 'Open'

<img width="336" alt="image" src="https://user-images.githubusercontent.com/82328870/208985005-4199a566-8424-4bf4-beea-d31a6bb42f0a.png">

9. If everything was done OK you will connect to AWC server 

<img width="492" alt="image" src="https://user-images.githubusercontent.com/82328870/208985280-a41774b1-b64a-49de-bf2d-5ed099fb2c27.png">

10. In the console type the following commands, what will install miniconda (other options are acceptable here too). And exit and reenter the AWS server

`$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

`$ sh ./Miniconda3-latest-Linux-x86_64.sh`

11. Create environment with python=3.7 by typing the following comands. Attention: other versions of python (for example, newer one, like 3.8) `pywren` won't run. 

`$ conda create --name jupyter python=3.7`

`$ conda activate jupyter`

`$ conda install jupyterlab`

`$ jupyter lab`

12. If eveything is done correctly, there will appear a link, which you can copy to browser and empty jupyter, runned on AWS, will open - congrats you did most of boring part!

<img width="500" alt="image" src="https://user-images.githubusercontent.com/82328870/208986157-43f6341e-bc9b-489d-a7c2-4ef7566f5c80.png">

## Further actions to run our project

1. Clone our git project, `numpywren` repo - unfortunetly, it is unavailable in `pip` and the serverless-straggler-mitigation repo to use matvec realization function from it

`git clone https://github.com/linglingec/HeisenbergFormNLAProject`

`git clone https://github.com/Vaishaal/numpywren`

`git clone https://github.com/vipgupta/serverless-straggler-mitigation`

2. Install the neede libraries from requirements

`pip install -r requirements.txt`
