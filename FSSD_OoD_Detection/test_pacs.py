import subprocess

dist = ['photo','art_painting','cartoon','sketch']
#method = ['baseline','odin','maha']
#method = ['fss']
method = ['de']
for m in method:
	for ind in dist:
    		for ood in dist:
        		f = open('./'+m+'/'+ind+'_'+ood+".txt", "w")
        		subprocess.call(['python', 'test_'+m+'_pacs.py', '--ind',ind,'--ood',ood,'--model_arch','resnet'],stdout=f)
        		print('############################\n'+'Method: '+m+'\n\t--> In-distribution:'+ind+'\n\t--> Out-Distribution:'+ood+'\nDone')


