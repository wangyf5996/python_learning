import subprocess

def getHtml(url):
	cmd = "phantomjs ./get_html.js '%s'" % url
	print 'cmd=', cmd
	stdout, stderr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
	#print stderr
	#print stdout
	return stdout

if __name__ == '__main__':
	html = getHtml('http://www.iwencai.com/stockpick/search?typed=0&preParams=&ts=1&f=1&qs=1&selfsectsn=&querytype=&searchfilter=&tid=stockpick&w=macd%E9%87%91%E5%8F%89')
	print html
