#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("deliveries.csv")


# In[3]:


df.head(2)


# In[4]:


df=df[(df.inning ==1) | (df.inning==2)]


# In[5]:


df.inning.unique()


# In[6]:


# choosing the teams


# In[7]:


# gt vs csk


# In[8]:


# t1 = bats first, t2 = bats second


# In[9]:


t1 = "Gujarat Titans"
t2 = "Chennai Super Kings"


# In[10]:


# choosing a match between both the teams where gt batted first and csk second


# In[11]:


df[(df.batting_team == t1) & (df.bowling_team == t2) & (df.inning == 1)].match_id.unique()


# In[12]:


mdf =df[df.match_id == 1370353]


# In[13]:


gt =mdf[mdf.inning ==1]
csk = mdf[mdf.inning ==2]


# In[14]:


# Outcomes- here we will handle runs and wickets separately instead of in same outcome like used by MAS 


# In[15]:


outcomes = [0,1,2,3,4,6]


# In[16]:


# These lists can be used to predict future scores, simulate innings, 
# or even determine which team is more likely 
# to win based on their scoring patterns and number of wickets lost.( so lets make the list)


# In[17]:


gt_outcomes = df[df.batting_team == t1].groupby("total_runs").size().reindex(outcomes, fill_value = 0)


# In[18]:


gt_wickets = df[df.batting_team ==t1]["is_wicket"].sum()


# In[19]:


csk_outcomes = df[df.batting_team ==t2].groupby("total_runs").size().reindex(outcomes, fill_value =0)


# In[20]:


csk_wickets = df[df.batting_team ==t2]["is_wicket"].sum()


# In[21]:


# now lets join the outcome and wickets together


# In[22]:


gt_outcomes = list(gt_outcomes) + [gt_wickets]


# In[23]:


csk_outcomes = list(csk_outcomes) + [csk_wickets]


# In[24]:


gt_outcomes


# In[25]:


csk_outcomes


# In[26]:


# Calculating Probabilities


# In[27]:


gt_outcomes_array = np.array(gt_outcomes)

csk_outcomes_array = np.array(csk_outcomes)


# In[28]:


# lets normalize it in order to get the probabilities


# In[29]:


gt_pb_ls = gt_outcomes_array / gt_outcomes_array.sum()


# In[30]:


csk_pb_ls = csk_outcomes_array / csk_outcomes_array.sum()


# In[31]:


gt_pb_ls


# In[32]:


gt_pb_list = np.cumsum(gt_pb_ls)


# In[33]:


csk_pb_list = np.cumsum(csk_pb_ls)


# In[34]:


# the cumulative sum shows the probability like 0.21 for 0, then 0.54 for 1,0 both and then 0.68 for 0,1,2 and so on.
# that is why we use cumsum to make out simulation smoother


# In[35]:


np.random.random()


# In[36]:


pred_runs = 0
pred_wks = 0
balls = 120
for i in range(balls):
    r_value = np.random.random()
    
    if r_value <= gt_pb_list[0]:
        outcome = 0
    elif r_value <= gt_pb_list[1]:
        outcome = 1
    elif r_value <= gt_pb_list[2]:
        outcome = 2
    elif r_value <= gt_pb_list[3]:
        outcome = 3 
    elif r_value <= gt_pb_list[4]:
        outcome = 4
    elif r_value <= gt_pb_list[5]:
        outcome = 6
    else:
        outcome = "W"    
    if outcome == "W":
        pred_wks += 1
        
        if pred_wks == 10:
            break
    else:
        pred_runs += outcome 


# In[37]:


print(f"Predicted Runs for Gujarat Titans: {pred_runs}")
print(f"Predicted Wickets for Gujarat Titans: {pred_wks}")


# In[38]:


# lets make a function where we can input a situational score, wkt, over and then can find out what will be the end score


# In[39]:


gt["over_ball"] = gt["over"] + gt["ball"] / 10


# In[40]:


gt.head()


# In[41]:


csk["over_ball"] = csk["over"] + csk["ball"]/10


# In[42]:


# lets make overball to number of ball like 19.1 overs as 115 balls


# In[43]:


over_ball = 19.1
over_no = int(over_ball)
ball_no = int((over_ball - over_no)* 10)
print("total_balls =" ,over_no*6 + ball_no)


# In[44]:


# Define the function to simulate Gujarat Titans' innings
def gt_inning_1(curr_runs, curr_wkts, curr_overs):
    pred_runs = curr_runs
    pred_wkts = curr_wkts
    over_ball = curr_overs
    
    # Split current overs into over number and ball number
    over_no = int(over_ball)
    ball_no = int((over_ball - over_no) * 10)
    
    # Calculate how many balls are left in the innings
    left_over_balls = 120 - (over_no * 6 + ball_no)
    
    for i in range(left_over_balls):
        # Generate random value for outcome simulation
        r_value = np.random.random()
        
        # Determine the outcome based on the random value
        if r_value <= gt_pb_list[0]:
            outcome = 0
        elif r_value <= gt_pb_list[1]:
            outcome = 1
        elif r_value <= gt_pb_list[2]:
            outcome = 2
        elif r_value <= gt_pb_list[3]:
            outcome = 3
        elif r_value <= gt_pb_list[4]:
            outcome = 4
        elif r_value <= gt_pb_list[5]:
            outcome = 6
        else:
            outcome = "W"  # Wicket
        
        # Update runs or wickets based on the outcome
        if outcome == "W":
            pred_wkts += 1
            # If 10 wickets fall, stop the innings
            if pred_wkts == 10:
                break
        else:
            pred_runs += outcome
        
        
    
    # Return the predicted total runs at the end of the innings
    return pred_runs
        


# In[45]:


gt_inning_1(190,4,19.1)


# In[46]:


# Now lets make it for the second innings that means csk chasing the target


# In[47]:


def csk_2nd_inning(curr_runs, curr_wkts, curr_overs, target):
    pred_runs = curr_runs
    pred_wkts = curr_wkts
    over_ball = curr_overs
    
    over_no = int(over_ball)
    ball_no = int((over_ball - over_no)*10)
    leftover_balls = 120 - (over_no*6 + ball_no)
    
    for i in range(leftover_balls):
        r_value = np.random.random()
        
        if r_value <= csk_pb_list[0]:
            outcome = 0
        elif r_value <= csk_pb_list[1]:
            outcome = 1
        elif r_value <= csk_pb_list[2]:
            outcome = 2
        elif r_value <= csk_pb_list[3]:
            outcome = 3
        elif r_value <= csk_pb_list[4]:
            outcome = 4
        elif r_value <= csk_pb_list[5]:
            outcome = 6
        else:
            outcome ="W"
            
        if outcome == "W":
            pred_wkts += 1
            
            if pred_wkts == 10:
                break
        else:
            pred_runs += outcome
            
        if pred_runs >= target:
            break
            
    return pred_runs        
        
        


# In[48]:


csk_2nd_inning(120, 0, 12.5, 140)


# ## runs prediction gt first inning

# In[49]:


curr_runs = 0
curr_wkts = 0
curr_overs = 0.0

gt_runs_pred = []
for i in range(len(gt)):
    curr_runs += gt.total_runs.iloc[i]
    curr_wkts += gt.is_wicket.iloc[i]
    curr_overs += gt.over_ball.iloc[i]
    prediction = gt_inning_1(curr_runs, curr_wkts, curr_overs)
    gt_runs_pred.append(prediction)


# In[50]:


sum(gt.total_runs)


# In[51]:


actual_score = sum(gt.total_runs)


# In[52]:


x_values = [i for i in range(len(gt))]
y_values = gt_runs_pred
plt.figure(figsize = (16,6))
plt.scatter(x_values, y_values, color = "red", alpha = 0.5, label = "pred")
plt.axhline(actual_score, ls = "--", color = "green", label = "actual" )
plt.ylim(0, actual_score + 40)
plt.title("GT Runs Prediction :" + str(actual_score), fontsize = 16 )
plt.xlabel("Ball no")
plt.ylabel("Runs")
plt.legend()
plt.show()


# ## runs prediction 2nds innings

# In[53]:


target = actual_score

curr_runs = 0
curr_wkts = 0
curr_overs = 0.0

csk_runs_pred = []

for i in range(len(csk)):
    curr_runs += csk.total_runs.iloc[i]
    curr_wkts += csk.is_wicket.iloc[i]
    curr_overs += csk.over_ball.iloc[i]
    
    prediction = csk_2nd_inning(curr_runs, curr_wkts, curr_overs, target)
    csk_runs_pred.append(prediction)


# In[54]:


csk_actual_score = sum(csk.total_runs)


# In[55]:


plt.figure(figsize = (16, 6))
plt.scatter([i for i in range(len(csk_runs_pred))], csk_runs_pred, alpha = 0.5, label = 'pred', color = 'red')
plt.ylim(0, csk_actual_score + 30)
plt.axhline(csk_actual_score, ls = '--', label = 'actual', color = 'green')
plt.title('Second Innings Runs - Prediction vs Actual (' + t2 + ': ' + str(csk_actual_score) +  ')', fontsize = 16)
plt.xlabel('Ball No')
plt.ylabel('Runs')
plt.legend()
plt.show()


# In[56]:


# to find out how much the error is


# In[57]:


np.mean([abs(i - csk_actual_score) for i in csk_runs_pred])


# # Now we will finally create a win prediction model

# Win prediction first innings

# In[60]:


win_cnt = 0
lose_cnt = 0
tie_cnt = 0

win_cnt_ls = []
lose_cnt_ls = []
tie_cnt_ls = []

gt_curr_runs = 0
gt_curr_wkts = 0
gt_curr_overs = 0

for i in range(len(gt)):
    gt_curr_runs += gt.total_runs.iloc[i]
    gt_curr_wkts += gt.is_wicket.iloc[i]
    gt_curr_overs += gt.over_ball.iloc[i]
    
    csk_curr_runs = 0
    csk_curr_wkts = 0
    csk_curr_overs = 0
    
    for j in range(100):
        gt_runs_prediction = gt_inning_1(curr_runs, curr_wkts, curr_overs)
        target = gt_runs_prediction
        
        csk_runs_prediction = csk_2nd_inning(curr_runs, curr_wkts, curr_overs, target)
        
        if csk_runs_prediction > target:
            win_cnt += 1
        elif csk_runs_prediction == target:
            tie_cnt += 1
        else:
            lose_cnt += 1
     
    win_cnt_ls.append(win_cnt)
    tie_cnt_ls.append(tie_cnt)
    lose_cnt_ls.append(lose_cnt)
    
    win_cnt = 0
    tie_cnt = 0
    lose_cnt = 0


# # Win prediction for csk 2nd inning

# In[64]:


csk_curr_runs = 0
csk_curr_wkts = 0
csk_curr_overs = 0.0
target = actual_score

for i in range(len(csk)):
    csk_curr_runs += csk.total_runs.iloc[i]
    csk_curr_wkts += csk.is_wicket.iloc[i]
    csk_curr_overs += csk.over_ball.iloc[i]
    
    for j in range(100):
        csk_runs_prediction = csk_2nd_inning(curr_runs, curr_wkts, curr_overs, target)
        
        if csk_runs_prediction > target:
            win_cnt += 1
        elif csk_runs_prediction == target:
            tie_cnt += 1
        else:
            lose_cnt += 1
            
    win_cnt_ls.append(win_cnt)
    tie_cnt_ls.append(tie_cnt)
    lose_cnt_ls.append(lose_cnt)
    
    win_cnt = 0
    tie_cnt = 0
    lose_cnt = 0
        


# In[65]:


len(win_cnt_ls), len(tie_cnt_ls), len(lose_cnt_ls)


# In[66]:


plt.figure()

x1_values = [i for i in range(len(win_cnt_ls))]
y1_values = win_cnt_ls

x2_values = [i for i in range(len(lose_cnt_ls))]
y2_values = lose_cnt_ls

x3_values = [i for i in range(len(tie_cnt_ls))]
y3_values = tie_cnt_ls

for i in range(10, len(mdf), 20):
    if i < len(mdf) - 10:
        plt.axvspan(i, i+10, ymin = 0, ymax = 100, alpha = 0.05, color='grey')
        
plt.axhline(y = 75, ls = '--', alpha = 0.3, c = 'grey')
plt.axhline(y = 50, ls = '--', alpha = 1, c = 'grey')
plt.axhline(y = 25, ls = '--', alpha = 0.3, c = 'grey')

plt.plot(x1_values, y1_values, color = 'orange', label = t2)
plt.plot(x2_values, y2_values, color = 'grey', label = 'Tie Value')
plt.plot(x3_values, y3_values, color = 'blue', label = t1)

plt.ylim(0, 100)
plt.yticks([0, 25, 50, 75, 100])

plt.title('Win Percentage Chart: ' + t1 + ' vs ' + t2, fontsize = 16)
plt.xlabel('Ball No')
plt.ylabel('Win %')
plt.legend()
plt.show()


# In[ ]:




