#include<iostream> 
using namespace std; 

class phdStudent{
    int x;
};

class InternalFullTime: public phdStudent{
    public:
    string ty="IFT";
    string name;
};

class InternalPartTime: public phdStudent{
    public:
    string ty="IPT";
    string name;
    
};

class ExternalPartTime: public phdStudent{
    public:
    string ty="EPT";
    string name;
};

int main(){
    
    InternalFullTime a;
    InternalPartTime b;
    ExternalPartTime c;
    string name,typ;
    for(int i=0; i<3;i++){
    cin>>name>>typ;
    if(typ.compare(a.ty)){
        a.name=name;
        cout<<a.name<<endl<<typ<<endl;
    }
    else if(typ.compare(b.ty)){
        b.name=name;
        cout<<b.name<<endl<<typ<<endl;
    }
    else if(typ.compare(c.ty)){
        c.name=name;
        cout<<c.name<<endl<<typ<<endl;
    }
    }


    return 0;
}
