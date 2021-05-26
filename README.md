<h1>session-aware</h1>
<h2>Introduction</h2>

<b>session-aware</b> is a Python-based framework for building and evaluating recommender systems (Python 3.5.x). It
implements a suite of state-of-the-art algorithms and baselines for session-based and session-aware recommendation.
<br><br>
Parts of the framework and its algorithms are based on code developed and shared by:
<ul>
    <li>
        Quadrana et al., Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks, RecSys 2017. <a
            href="https://github.com/mquad/hgru4rec">(Original Code).</a>
    </li>
    <li>
        Ruocco et al., Inter-session modeling for session-based recommendation, DLRS 2017. <a
            href="https://github.com/rainmilk/ieee-is-ncsf">(Original Code).</a>
    </li>
    <li>
        Ying et al., Sequential recommender system based on hierarchical attention network, IJCAI 2018. <a
            href="https://github.com/chenghu17/Sequential_Recommendation/tree/master/SHAN">(Original Code).</a>
    </li>
    <li>
        Liang et al., Neural cross-session filtering: Next-item prediction under intra- and inter-session context, IEEE Intelligent Systems 2019. <a
            href="https://github.com/rainmilk/ieee-is-ncsf">(Original Code).</a>
    </li> 
    <li>
        Phuong et al., Neural session-aware recommendation, IEEE Access 2019. <a
            href="https://github.com/thanhtcptit/RNN-for-Resys">(Original Code).</a>
    </li>    
    <li>
        Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009. <a
            href="https://github.com/hidasib/GRU4Rec/blob/master/baselines.py">(Original Code).</a>
    </li>
    <li>
        Mi et al., Context Tree for Adaptive Session-based Recommendation, 2018. (Code shared by the authors).
    </li>
    <li>
        Hidasi et al., Recurrent Neural Networks with Top-k Gains for Session-based Recommendations, CoRR
        abs/1706.03847, 2017. <a href="https://github.com/hidasib/GRU4Rec">(Original Code).</a>
    </li>
    <li>
        Liu et al., STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation, KDD
        2018. <a href="https://github.com/uestcnlp/STAMP">(Original Code).</a>
    </li>
    <li>
        Li et al., Neural Attentive Session-based Recommendation, CIKM 2017. <a
            href="https://github.com/lijingsdu/sessionRec_NARM">(Original Code).</a>
    </li>
    <li>
        Yuan et al., A Simple but Hard-to-Beat Baseline for Session-based Recommendations, CoRR
        abs/1808.05163, 2018. (Code shared by the authors).
    </li>
    <li>
        Wu et al., Session-based recommendation with graph neural networks, AAAI, 2019. <a
            href="https://github.com/CRIPAC-DIG/SR-GNN">(Original Code).</a>
    </li>
    <li>
        Wang et al., A collaborative session-based recommendation approach with parallel memory modules, SIGIR, 2019. <a
            href="https://github.com/wmeirui/CSRM_SIGIR2019">(Original Code).</a>
    </li>
    <li>
        Rendle et al., Factorizing Personalized Markov Chains for Next-basket Recommendation. WWW 2010. <a
            href="https://github.com/rdevooght/sequence-based-recommendations/blob/master/factorization/fpmc.py">(Original
        Code).</a>
    </li>
    <li>
        Kabbur et al., FISM: Factored Item Similarity Models for top-N Recommender Systems, KDD 2013. <a
            href="https://github.com/rdevooght/sequence-based-recommendations/blob/master/factorization/fism.py">(Original
        Code).</a>
    </li>
    <li>
        He and McAuley. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation.
        CoRR abs/1609.09152, 2016. <a
            href="https://github.com/rdevooght/sequence-based-recommendations/blob/master/factorization/fossil.py">(Original
        Code).</a>
    </li>
</ul>


<h2>Requirements</h2>
To run session-aware, the following libraries are required:
<ul>
    <li>Anaconda 4.X (Python 3.5)</li>
    <li>Pympler</li>
    <li>NumPy</li>
    <li>SciPy</li>
    <li>BLAS</li>
    <li>Sklearn</li>
    <li>Dill</li>
    <li>Pandas</li>
    <li>Theano</li>
    <li>Pyyaml</li>
    <li>CUDA</li>
    <li>Tensorflow</li>
    <li>Theano</li>
    <li>Psutil</li>
    <li>Scikit-learn</li>
    <li>Tensorflow-gpu</li>
    <li>NetworkX</li>
    <li>Certifi</li> 
    <li>NumExpr</li>
    <li>Pytable</li>
    <li>Python-dateutil</li>
    <li>Pytz</li>
    <li>Six</li>
    <li>Keras</li>
    <li>Scikit-optimize</li>
    <li>Python-telegram-bot</li>
</ul>

<h2>Installation</h2>
<ol>
    <li>Download and Install Anaconda (https://www.anaconda.com/distribution/)</li>
    <li>Unzip the file, and in the main folder, run the following commands:
        <ol>
            <li><code>conda install --yes --file requirements_conda.txt</code></li>
            <li><code>pip install -r requirements_pip.txt</code></li>      
        </ol>
    </li>
</ol>
<h2>How to Run It</h2>
<ol>
    <li><h5>Dataset preprocessing</h5>
    <ol>
        <li>
            Unzip and add any dataset file to the data folder, i.e., events.hdf will then be in the folder
            data/retairocket/raw
        </li>
        <li>
            Open and edit any configuration file in the folder conf/preprocess/.. to configure the preprocessing method and parameters.
            <ul>
                <li>
                    See, e.g., conf/preprocess/session-aware/retailrocket_window.yml for an example with comments.
                </li>
            </ul>
        </li>
        <li>
            Run a configuration with the following command: </br>
            <code>
                python run_preprocesing.py conf/preprocess/session-aware/retailrocket_window.yml
            </code>
        </li>
    </ol>
    </li>
    <li><h5>Run experiments using the configuration file</h5>
    <ol>
        <li>
            Create folders conf/in and conf/out. Configure a configuration file <b>*.yml</b> and put it into the folder named conf/in. Examples of configuration
            files are listed in the conf folder. It is possible to configure multiple files and put them all in
            the conf/in folder. When a configuration file in conf/in has been executed, it will be moved to the folder conf/out.
        </li>
        <li>
            Run the following command from the main folder: </br>
            <code>
                python run_config.py conf/in conf/out
            </code></br>
            If you want to run a specific configuration file, run the following command:</br>
            <code>
                python run_config.py conf/example_next.yml
            </code>
        </li>    
        <li>
            Results will be displayed and saved to the results folder as config.
        </li>
    </ol>
    </li>
</ol>


<h2>How to Configure It</h2>
<b>Start from one of the examples in the conf folder.</b>
<h3>Essential Options</h3>
<div>
<div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="12%" scope="col"> Entry</th>
            <th width="16%" class="conf" scope="col">Example</th>
            <th width="72%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td>type</td>
            <td>window</td>
            <td>Values: single (one single training-test split), window (sliding-window protocol), opt (parameters
                optimization).
            </td>
        </tr>
        <tr>
            <td>evaluation</td>
            <td>evaluation_user_based</td>
            <td>Values: <b>for session-aware evaluation:</b> evaluation_user_based (evaluation in term of the next item and in terms of the remaining items of the sessions),  <b>for session-based evaluation:</b> evaluation (evaluation in term of the next item), evaluation_last (evaluation in term of the last item of the session), evaluation_multiple (evaluation in terms of the remaining items of the sessions).
            </td>
        </tr>
        <tr>
            <td scope="row">slices</td>
            <td>5</td>
            <td>Number of slices for the window protocol.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">opts</td>
            <td>opts: {sessions_test: 10}</td>
            <td>Number of sessions used as a test during the optimization phase.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">metrics</td>
            <td>-class: accuracy.HitRate<br>
                length: [5,10,15,20]
            </td>
            <td>List of accuracy measures (HitRate, MRR, Precision, Recall, MAP, Coverage, Popularity,
                Time_usage_training, Time_usage_testing, Memory_usage).
                If you want to save the files with the recommedation lists use the option: <br>
                <code> - class: saver.Saver<br>
                    length: [50]</code>
                It's possible to use the saved recommendations using the ResultFile class.
            </td>
        </tr>
        <tr>
            <td scope="row">opts</td>
            <td>opts: {sessions_test: 10}</td>
            <td>Number of session used as a test during the optimization phase.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">optimize</td>
            <td> class: accuracy.MRR <br>
                length: [20]<br>
                iterations: 100 #optional
            </td>
            <td>Measure to which optimize the parameters.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">algorithms</td>
            <td>-</td>
            <td>See the configuration files in the conf folder for a list of the
                algorithms and their parameters.<br>
            </td>
        </tr>
    </table>
</div>
</div>
