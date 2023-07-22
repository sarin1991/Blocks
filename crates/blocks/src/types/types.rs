use std::marker::{Sync,Send};
pub trait BlockConfig {}

pub trait BlockType {
    type Config:BlockConfig;
}

pub trait BlockTypeAllocate : BlockType {
    fn allocate(config:&Self::Config) -> Self;
}

pub trait ViewRepr : Sync + Send {
    type View<'a> : Sync + Send where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a>;    
}

pub trait ValidateRepr : ViewRepr + BlockType {
    fn validate<'a>(config: &Self::Config,view: &Self::View<'a>) -> bool;
}

pub trait ViewMutRepr : ViewRepr {
    type ViewMut<'a> : ViewRepr<View<'a> = Self::View<'a>>
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a>;
}