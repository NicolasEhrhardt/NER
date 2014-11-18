We've set things up so that it just works out of the box with Eclipse and Ant. 


ECLIPSE:
Go to File->New->New Project (Not Java Project). Select Java Project from existing Ant Buildfile and then select the build.xml in this folder. 
To set up a run configuration for NER, right-click NER.java in the project explorer and select "Run As" -> "Run Configuration". 
Under the Main tab, type "NER" for project and "NER" for main class. Then, proceed to the arguments tab and type "../data/train ../data/dev -print" for program arguments and "-Xmx1G" for VM arguments without the quotations. 
Then, click "Apply" and "Run" buttons. This will run your program.
 
ANT:
we've provided a basic build.xml file for use with ant.  Just call:

$ ant



COMMAND LINE:
If you want to develop on the command line, use the following commands:
$ mkdir classes
$ ant
$ java -Xmx1g -cp classes cs224n.deep.NER ../data/train ../data/dev -print
