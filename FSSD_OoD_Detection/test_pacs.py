import subprocess

dist = ['photo','art_painting','cartoon','sketch']


for ind in dist:
    for ood in dist:
        f = open('./odin/'+ind+'_'+ood+".txt", "w")
        subprocess.call(['python', 'test_odin_pacs.py', '--ind',ind,'--ood',ood,'--model_arch','resnet'],stdout=f)
        print(ind+'_'+ood+'Done')
