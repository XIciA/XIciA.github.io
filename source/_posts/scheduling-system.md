---
title: 智能排班系统
toc: true
date: 2023-4-16 00:00:01
tags: 智能设计
categories: 技术
thumbnail: https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/schedulesystemAIlogo.png
---

---

本项目采用了前后端分离框架。使用html，java，css语言进行前端页面搭建。使用php连接前端页面和MySQL数据库。使用Python完成后端排班运算过程，搭建Flask框架实现前后端数据连接。

<!-- more -->







---
# 前言
## 背景
随着新技术的发展，劳动力管理正在向智能化升级。管理者需要解决员工技能匹配、评估劳动力需求、提高员工效率、降低用工成本等问题。
## 目标
本项目致力于将劳动力与业务需求最优化匹配，综合门店管理规则，客户流量波动，员工偏好等因素，将合适数量的员工在合适的时间放在合适的位置上。
## 整体解决思路
本项目采用了前后端分离框架。使用html，java，css语言进行前端页面搭建。使用php连接前端页面和MySQL数据库。使用Python完成后端排班运算过程，搭建Flask框架实现前后端数据连接。
![整体解决思路图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/整体解决思路图.png)
## 解决方案
解决排班问题时，基于0-1整数规划，变量由周、日、员工、时间段决定。根据门店规则，员工偏好和预测客流量确定约束条件，根据排班总时长和偏好满足程度确定目标方程，建立求解排班结果的MIP模型。使用分支定界算法求解该模型，得到最终排班结果。


---
# 创意描述
(1)使用前后端分离来实现智能排班系统，提高了系统的可扩展性和灵活性。  
(2)建立了求解排班结果的MIP混合整数规划模型，保证了系统在不同场景下的适用性。  
(3)采用分支界定算法进行求解，保证了程序的正确性和时间效率。  
(4)前端页面的下拉菜单，多选框等元素中的可选项能随着数据库中的元素改变而改变。    
(5)提供了Excle表格导出功能，让用户更方便地查看和使用排班数据。  
(6)在完成赛题所有要求的情况下，制定了夜班规则和限流规则，使该项目更加贴近实际场景。  
(7)本项目自创方程组，以多个变量为单位进行遍历，完美地解决员工连续工作时长限制问题。  



---
# 功能简介
## 登录功能
用户可以通过已经注册的信息进行登录，从而进入“翼排班”智能排班系统。
## 排班与查看
### 生成排班表
在完成所有信息的管理之后，用户可点击“一键排班”按钮实现排班表的生成。
![生成排班表示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/生成排班表示意图.png)
### 查看排班表
生成排班表后，用户可根据自己的需求，查看不同门店和不同日期的排班表，同时可以选择周视图、日视图、按员工分组、按岗位分组和按技能分组等方式进行查看。
![查看排班表示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/查看排班表示意图.png)
### 导出排班表
用户可以根据需求导出在不同查看方式下的排班表，导出格式为Excel。
![导出Excel示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/导出Excel示意图.png)
### 修改排班表
用户可根据需求修改排班表，可通过“指定时间段”按钮来修改特定员工上班的时间段，可通过“指定班次”按钮来调整员工在特定时间段的班次。
![指定时间段示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/指定时间段示意图.png)
![指定班次示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/指定班次示意图.png)
## 信息管理功能
### 门店信息管理
在门店管理界面，可以通过输入门店的各种数据将门店信息上传至数据库。
![门店信息管理示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/门店信息管理示意图.png)
### 员工信息管理
在员工信息管理界面，可以通过输入员工的各种数据将员工信息上传至数据库，包括员工编号、员工姓名、工作日偏好、工作时间偏好、最大工作时间等。
![员工信息管理示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/员工信息管理示意图.png)
### 自定义排班规则管理
选择自定义规则应用门店后，可设置开店规则、关店规则、客流规则、夜班规则、限流规则。 在满足赛题的所有要求的情况下，本作品自行拟定了夜班规则和限流规则。
![自定义排班规则示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/自定义排班规则示意图.png)
### 固定排班规则管理
固定规则包括门店营业时间规则、工作时长规则、休息时间段等。满足赛题的所有要求。
![固定规则示意图](https://vaporfigurebed.oss-cn-beijing.aliyuncs.com//blogimg/固定规则示意图.png)
### 数据查看与修改
在完成信息的输入后，本项目支持对业务预测数据，员工数据，门店数据的查看与修改。


---
# 特色综述
本项目采用前后端分离框架。建立MIP混合整数规划模型，采用分支界定算法求解。提供了Excle表格导出功能，更方便用户查看和使用排班数据。使用php连接前端页面和MySQL数据库，搭建Flask框架实现前后端数据连接。

---
# 开发工具与技术
| 应用模块    | 开发工具   | 技术                       |
|:----------:|:--------:|:--------------------------:|
| 前端网页  |  VScode  | Html, Css, JavaScript       |
| 数据传输    |  VScode  | MySQL, php, Flask           |
| 后端算法  |  VScode  | Python, Cplex, MIP          |


---
# 应用对象
本web应用适用于企业管理者和人力资源负责人，以及所有需要为员工安排工作的人员或企业。

---
# 应用环境
智能排班系统是新兴技术，近年来在医疗保健、零售、酒店、餐饮等行业广泛应用。全球智能排班系统市场规模不断扩大，未来仍将持续增长。在各行各业都有着广泛的应用前景。

---
# 应用结语
经过测试，本项目能够以最优整数目标值与剩余最优节点目标值之间的差距值为0.028的条件下，在3.22s内，适应客流变化，并满足所有员工偏好及门店规则，生成排班表。并提供查看，筛选，导出等功能。


